"""
Exp 9 (RLMC + Reranker) Full Dev Set Evaluation
================================================
Mirrors eval_exp15.py. Pipeline:
  Stage 1+2 : RLMC agent traverses KG with adaptive beam (TIGHT/MEDIUM/LOOSE)
  Stage 3   : Cross-encoder reranker (exp9_reranker_final.pt) picks the best entity

Metrics reported:
  Hit@1  : reranker's top-1 entity is in gold answer set
  Hit@N  : any entity in the candidate set is in gold answer set (raw recall)
"""
import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from utils.sparql_parser import find_reasoning_path


# ──────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────

class Exp9EvalPipeline:
    def __init__(self, ckpt=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Init] Starting Exp 9 (RLMC) Evaluation Pipeline (Device: {self.device})")

        data_dir = os.path.join(ROOT, "data/processed_entity")
        self.rel2id = torch.load(os.path.join(data_dir, "relation2id.pt"), map_location="cpu")
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, "domain2id.pt"), map_location="cpu")
        self.id2dom = {v: k for k, v in self.dom2id.items()}

        # LMDB KG
        lmdb_path = os.path.join(ROOT, "data/processed_kg/augmented_kg_lmdb")
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        print("[Init] KG loaded via LMDB.")

        # Entity names
        print("[Init] Loading Entity Names...")
        self.mid2name = json.load(open(os.path.join(ROOT, "data/master_mid2name.json"), encoding="utf-8"))

        # RLMC agent
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.agent = RLConstraintAgent(base_model).to(self.device)

        if ckpt is None:
            ckpt = os.path.join(ROOT, "checkpoints/exp9_rlmc_epoch_9.pt")

        if not os.path.exists(ckpt):
            print(f"[Warning] Checkpoint not found at {ckpt}.")
        else:
            print(f"[Init] Loading RLMC weights from {ckpt}...")
            self.agent.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.agent.eval()

        # Cross-encoder reranker
        reranker_ckpt = os.path.join(ROOT, "checkpoints/exp9_reranker_final.pt")
        print(f"[Init] Loading reranker from {reranker_ckpt}...")
        self.reranker = RobertaForSequenceClassification.from_pretrained(
            "roberta-large", num_labels=1).to(self.device)
        self.reranker.load_state_dict(
            torch.load(reranker_ckpt, map_location=self.device, weights_only=False),
            strict=False)
        self.reranker.eval()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        print("[Init] Pipeline Ready.\n")

    @torch.no_grad()
    def _run_agent(self, question, topic_mid):
        """Run RLMC traversal. Returns final entity set."""
        inputs = self.tokenizer(
            question, return_tensors="pt", padding=True,
            truncation=True, max_length=128).to(self.device)

        action_logits, _, rel_logits, dom_logits = self.agent(
            inputs["input_ids"], inputs["attention_mask"])

        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]

        current = {topic_mid}
        for h in range(4):
            action = torch.argmax(action_logits[0, h]).item()
            if action == 3:
                break
            if action == 0:   # TIGHT
                active = [self.id2rel[torch.argmax(rel_logits[0, h]).item()]]
            elif action == 1: # MEDIUM
                active = [self.id2rel[rid]
                          for rid in torch.topk(rel_logits[0, h], 5).indices.tolist()]
            else:              # LOOSE
                active = [r for r in self.id2rel.values() if domain_name in r]

            nxt = set()
            with self.env.begin() as txn:
                for ent in current:
                    f_data = txn.get(f"f:{ent}".encode())
                    if f_data:
                        for r, tgt in pickle.loads(f_data):
                            if r in active: nxt.add(tgt)
                    b_data = txn.get(f"b:{ent}".encode())
                    if b_data:
                        for r, src in pickle.loads(b_data):
                            if r in active: nxt.add(src)
            if not nxt:
                break
            current = nxt

        return current

    @torch.no_grad()
    def _rerank(self, question, candidates, batch_size=32):
        """Score candidates with cross-encoder. Returns top-1 MID."""
        if not candidates:
            return None
        named = [(mid, self.mid2name.get(mid, "Unknown"))
                 for mid in candidates if mid in self.mid2name]
        if not named:
            return min(candidates)   # fallback: lexicographic

        scored = []
        for i in range(0, len(named), batch_size):
            batch = named[i:i+batch_size]
            mids_b = [m for m, _ in batch]
            names_b = [n for _, n in batch]
            enc = self.tokenizer(
                [question] * len(batch), names_b,
                padding=True, truncation=True,
                max_length=128, return_tensors="pt").to(self.device)
            logits = self.reranker(**enc).logits.squeeze(-1)
            scored.extend(zip(mids_b, logits.cpu().tolist()))

        return max(scored, key=lambda x: x[1])[0]


# ──────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────

def evaluate_exp9(split="dev", limit=None, ckpt=None):
    print(f"\n[Eval] Starting Exp 9 Evaluation on {split} set...")
    pipeline = Exp9EvalPipeline(ckpt=ckpt)

    data_path = os.path.join(ROOT, f"data/cwq_{split}.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        path = find_reasoning_path(item["sparql"])
        if not path:
            continue
        te = path[0][0].replace("ns:", "")
        gold = set(a["answer_id"].replace("ns:", "")
                   for a in item.get("answers", []) if a.get("answer_id"))
        if not gold:
            continue
        samples.append({"q": item["question"], "te": te, "gold": gold})
        if limit and len(samples) >= limit:
            break

    print(f"[Eval] Loaded {len(samples)} valid samples.")

    stats = {"hit1": 0, "hit_n": 0, "total_ents": 0, "count": 0, "dead_ends": 0}

    for s in tqdm(samples):
        candidates = pipeline._run_agent(s["q"], s["te"])

        stats["count"] += 1
        stats["total_ents"] += len(candidates)
        if not candidates:
            stats["dead_ends"] += 1

        hit_n = any(mid in s["gold"] for mid in candidates)
        if hit_n:
            stats["hit_n"] += 1

        best = pipeline._rerank(s["q"], candidates)
        if best and best in s["gold"]:
            stats["hit1"] += 1

    n = stats["count"] or 1
    hit1_pct  = stats["hit1"]  / n * 100
    hitn_pct  = stats["hit_n"] / n * 100
    avg_ents  = stats["total_ents"] / n

    print("\n" + "=" * 40)
    print(f"EXP 9 EVALUATION RESULTS ({split.upper()})")
    print("=" * 40)
    print(f"Hit@1 Accuracy:   {hit1_pct:.2f}%")
    print(f"Hit@N Accuracy:   {hitn_pct:.2f}% (Recall after traversal)")
    print(f"Avg Entities:     {avg_ents:.2f}")
    print(f"Dead Ends:        {stats['dead_ends']} / {stats['count']}")
    print(f"Total Samples:    {stats['count']}")
    print("=" * 40)

    out_path = os.path.join(ROOT, f"inference_pipeline/results_exp9_{split}.json")
    with open(out_path, "w") as f:
        json.dump({
            "hit1":     hit1_pct,
            "hit_n":    hitn_pct,
            "avg_ents": avg_ents,
            "samples":  stats["count"]
        }, f, indent=4)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ckpt",  type=str, default=None)
    args = parser.parse_args()
    evaluate_exp9(split=args.split, limit=args.limit, ckpt=args.ckpt)
