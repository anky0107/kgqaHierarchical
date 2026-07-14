"""
Exp 23: Two-Pass LMDB Verification
====================================

NO TRAINING REQUIRED. Post-processing step applied after Stage 3.

WHY THIS MATTERS
----------------
Stage 3 sometimes picks a plausible-sounding entity that is actually
UNREACHABLE from the topic entity via the predicted relation path.
Example: Stage 3 ranks "John Smith (politician)" first, but the traversal
path is film.film.directed_by → film.director.film, which can only reach
people who actually directed films. The correct answer "John Smith (director)"
is ranked second.

Fix: after Stage 3 picks the top-1 candidate, verify it is reachable
from the topic entity via the gold path in LMDB. If not, fall back to
rank-2 and verify that. Continue down the ranked list until a reachable
entity is found or the list is exhausted.

This costs exactly one LMDB path-existence check per question at inference
and requires no retraining of any model.

WHAT IS A VALID PATH CHECK
---------------------------
Given:
  - topic entity MID  (start node)
  - predicted answer MID (end node)
  - predicted relation sequence (r_1, r_2, ..., r_h)

Check: does a path exist in LMDB:
  topic_entity --r_1--> e_1 --r_2--> ... --r_h--> predicted_answer

This is NOT a full graph search. It is a targeted hop-by-hop lookup:
  hop 1: does (topic_entity, r_1, ?) exist → get e_1 set
  hop 2: does any e_1 --r_2--> predicted_answer exist?

If the sequence passes all hops → entity is reachable → accept.
If any hop fails → try next ranked candidate.

EXPECTED GAIN: +4–6% Hit@1
This is most effective when:
  - The correct answer is in the top-5 of Stage 3 (i.e. recall is fine
    but Stage 3 ranking is wrong)
  - The incorrect top-1 is genuinely unreachable from the topic entity

HOW TO RUN
----------
  python train/exp23_twopass_verify.py

  Optional flags:
    --cds_json   path to CDS eval JSON with path + mid fields
    --kg_path    path to augmented_kg.pt
    --s3_ckpt    path to Stage 3 checkpoint
    --top_k      how many candidates to verify before giving up (default: 5)

OUTPUT
------
  metrics/exp23_twopass_verify.csv  — Hit@1 before and after verification
"""

import os, sys, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)


# ─────────────────────────────────────────────────────────────
#  Path existence check using LMDB-style kg dict
# ─────────────────────────────────────────────────────────────

def path_exists(topic_mid: str, answer_mid: str,
                rel_sequence: list, kg: dict) -> bool:
    """
    Check whether a directed relation path exists in the KG from
    topic_mid to answer_mid following rel_sequence.

    Parameters
    ----------
    topic_mid    : starting entity Freebase MID
    answer_mid   : target entity Freebase MID
    rel_sequence : list of relation strings, e.g.
                   ["film.film.directed_by", "film.director.film"]
    kg           : dict with 'forward' key mapping
                   MID -> [(relation_str, target_MID), ...]

    Returns True if the path exists, False otherwise.

    Single-hop: rel_sequence = ["r1"]
      Check: answer_mid in {t for (r, t) in kg['forward'][topic_mid] if r == "r1"}

    Two-hop: rel_sequence = ["r1", "r2"]
      Step 1: get intermediate entities reachable via r1 from topic_mid
      Step 2: check if answer_mid is reachable via r2 from any intermediate

    Handles forward edges only. Extend to backward if your KG needs it.
    """
    if not rel_sequence or not topic_mid or not answer_mid:
        return False

    forward = kg.get("forward", {})
    current_set = {topic_mid}

    for i, rel in enumerate(rel_sequence):
        next_set = set()
        for mid in current_set:
            for edge_rel, target in forward.get(mid, []):
                if edge_rel == rel:
                    next_set.add(target)
        if not next_set:
            return False
        # On the final hop, check if answer_mid is reachable
        if i == len(rel_sequence) - 1:
            return answer_mid in next_set
        current_set = next_set

    return False


def parse_path_str(path_str: str) -> list:
    """
    Parse a stored path string into a list of relation strings.
    Supports space-separated or pipe-separated formats.

    "film.film.directed_by film.director.film"
      -> ["film.film.directed_by", "film.director.film"]
    """
    if not path_str or not path_str.strip():
        return []
    sep    = "|" if "|" in path_str else " "
    tokens = [t.strip() for t in path_str.split(sep) if t.strip()]
    # Filter out integer tokens (some datasets store IDs not strings)
    return [t for t in tokens if not t.isdigit()]


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class VerifyDataset(torch.utils.data.Dataset):
    """
    Reads CDS eval JSON. For verification to work, each item needs:
      - item['topic_mid']  : MID of the topic entity (start of traversal)
      - item['path']       : relation sequence string
      - candidate['mid']   : MID of the candidate entity

    If these fields are absent, verification falls back to accepting
    Stage 3's top-1 without checking (same as baseline).
    """
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp23 Dataset] {len(self.samples)} samples  "
              f"from {os.path.basename(json_path)}")

        # Warn if key fields are missing
        n_no_topic = sum(1 for s in self.samples if not s.get("topic_mid"))
        n_no_path  = sum(1 for s in self.samples if not s.get("path"))
        n_no_mid   = sum(1 for s in self.samples
                        for c in s["candidates"] if not c.get("mid"))
        if n_no_topic > 0:
            print(f"[Exp23 WARNING] {n_no_topic} items missing 'topic_mid' "
                  "— verification will be skipped for these.")
        if n_no_path > 0:
            print(f"[Exp23 WARNING] {n_no_path} items missing 'path' "
                  "— verification will be skipped for these.")
        if n_no_mid > 0:
            print(f"[Exp23 WARNING] {n_no_mid} candidates missing 'mid' "
                  "— those candidates cannot be verified.")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_passthrough(batch):
    return batch


# ─────────────────────────────────────────────────────────────
#  Stage 3 scoring
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def score_candidates(model, tok, question: str,
                     candidate_names: list,
                     device: torch.device,
                     max_length: int = 192) -> torch.Tensor:
    """Score all candidates for one question. Returns [N] logits."""
    all_qs = [question] * len(candidate_names)
    enc = tok(all_qs, candidate_names,
              padding=True, truncation=True,
              max_length=max_length,
              return_tensors="pt").to(device)
    return model(**enc).logits.squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, dataset: VerifyDataset,
             kg: dict, device: torch.device,
             mode: str = "baseline",
             top_k: int = 5,
             max_length: int = 192) -> dict:
    """
    mode = "baseline"  : use Stage 3 top-1 directly
    mode = "verified"  : verify top-k candidates, pick first reachable

    Returns dict with hit1, verify_triggered, verify_helped, verify_hurt.
    """
    model.eval()
    hits              = 0
    total             = 0
    verify_triggered  = 0   # questions where verification changed the answer
    verify_helped     = 0   # verification changed answer AND it became correct
    verify_hurt       = 0   # verification changed answer AND it became wrong
    no_mid_skipped    = 0   # questions skipped because no MID fields

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc=f"Eval [{mode}]", leave=False):
        item       = batch[0]
        q          = str(item["question"])
        cands      = item["candidates"]
        topic_mid  = item.get("topic_mid", "")
        rel_seq    = parse_path_str(item.get("path", ""))

        if not cands:
            continue
        gold_idx = next((i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None:
            continue

        names  = [str(c.get("name", "")) for c in cands]
        logits = score_candidates(model, tok, q, names, device, max_length)
        ranked = torch.argsort(logits, descending=True).tolist()

        if mode == "baseline":
            pred_idx = ranked[0]

        else:  # verified
            can_verify = bool(topic_mid and rel_seq
                              and any(c.get("mid") for c in cands))

            if not can_verify:
                # Cannot verify — fall back to baseline top-1
                pred_idx = ranked[0]
                if not any(c.get("mid") for c in cands):
                    no_mid_skipped += 1
            else:
                pred_idx = ranked[0]  # default: accept top-1

                for rank_pos in range(min(top_k, len(ranked))):
                    cand_idx  = ranked[rank_pos]
                    cand_mid  = cands[cand_idx].get("mid", "")
                    if not cand_mid:
                        continue    # no MID → skip verification for this candidate

                    reachable = path_exists(topic_mid, cand_mid, rel_seq, kg)

                    if reachable:
                        pred_idx = cand_idx
                        if rank_pos > 0:
                            # Verification changed the answer
                            verify_triggered += 1
                            old_correct = (ranked[0] == gold_idx)
                            new_correct = (pred_idx  == gold_idx)
                            if new_correct and not old_correct:
                                verify_helped += 1
                            elif old_correct and not new_correct:
                                verify_hurt   += 1
                        break
                else:
                    # No reachable candidate found in top_k — keep top-1
                    pass

        if pred_idx == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    return {
        "hit1":             hit1,
        "hits":             hits,
        "total":            total,
        "verify_triggered": verify_triggered,
        "verify_helped":    verify_helped,
        "verify_hurt":      verify_hurt,
        "no_mid_skipped":   no_mid_skipped,
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp23: Two-pass LMDB verification")
    parser.add_argument("--cds_json",   default=None,
                        help="CDS eval JSON (default: data/exp16_cds_dev.json)")
    parser.add_argument("--kg_path",    default=None,
                        help="KG path (default: data/processed_kg/augmented_kg.pt)")
    parser.add_argument("--s3_ckpt",    default=None,
                        help="Stage 3 checkpoint (default: checkpoints/exp16v2_s3_cross.pt)")
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Max candidates to verify before giving up (default: 5)")
    parser.add_argument("--max_length", type=int, default=192,
                        help="Tokenizer max length (default: 192)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp23] Device: {device}")

    cds_json = args.cds_json or os.path.join(ROOT, "data/exp16_cds_dev.json")
    kg_path  = args.kg_path  or os.path.join(ROOT, "data/processed_kg/augmented_kg.pt")
    s3_ckpt  = args.s3_ckpt  or os.path.join(ROOT, "checkpoints/exp16v2_s3_cross.pt")

    for path, label in [(cds_json, "CDS JSON"), (kg_path, "KG")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # ── Load KG ───────────────────────────────────────────────────────────────
    print(f"[Exp23] Loading KG ...")
    kg = torch.load(kg_path, map_location="cpu")
    print(f"[Exp23] KG loaded: {len(kg.get('forward', {}))} forward entities")

    # ── Load Stage 3 ──────────────────────────────────────────────────────────
    model_name = "BAAI/bge-reranker-base"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name).to(device)
    if os.path.exists(s3_ckpt):
        model.load_state_dict(torch.load(s3_ckpt, map_location=device))
        print(f"[Exp23] Loaded Stage 3: {s3_ckpt}")
    else:
        print(f"[Exp23] WARNING: {s3_ckpt} not found, using pretrained BGE.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VerifyDataset(cds_json)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"\n[Exp23] Running baseline (no verification) ...")
    base = evaluate(model, tok, dataset, kg, device,
                    mode="baseline", top_k=args.top_k,
                    max_length=args.max_length)

    # ── Verified ──────────────────────────────────────────────────────────────
    print(f"\n[Exp23] Running two-pass verification (top_k={args.top_k}) ...")
    veri = evaluate(model, tok, dataset, kg, device,
                    mode="verified", top_k=args.top_k,
                    max_length=args.max_length)

    # ── Report ────────────────────────────────────────────────────────────────
    delta = veri["hit1"] - base["hit1"]
    print(f"\n[Exp23] Results:")
    print(f"  Baseline Hit@1   : {base['hit1']:.4f}")
    print(f"  Verified Hit@1   : {veri['hit1']:.4f}  (delta: {delta:+.4f})")
    print(f"  Verify triggered : {veri['verify_triggered']} questions")
    print(f"  Verify helped    : {veri['verify_helped']} (wrong→correct)")
    print(f"  Verify hurt      : {veri['verify_hurt']}  (correct→wrong)")
    print(f"  No MID skipped   : {veri['no_mid_skipped']} questions")

    metrics_dir  = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp23_twopass_verify.csv")
    with open(metrics_path, "w") as f:
        f.write("mode,hit1,verify_triggered,verify_helped,verify_hurt\n")
        f.write(f"baseline,{base['hit1']:.4f},0,0,0\n")
        f.write(f"verified,{veri['hit1']:.4f},"
                f"{veri['verify_triggered']},"
                f"{veri['verify_helped']},"
                f"{veri['verify_hurt']}\n")
    print(f"  Results written to {metrics_path}")

    if veri["no_mid_skipped"] == veri["total"]:
        print("\n[Exp23] CRITICAL: All questions skipped verification because "
              "candidates have no 'mid' field. Add MID fields to your CDS "
              "JSON for this experiment to have any effect.")
    elif delta < 0.005:
        print("\n[Exp23] NOTE: Small gain. Check 'verify_triggered' count. "
              "If it is near zero, most Stage 3 top-1 candidates are already "
              "reachable, meaning the ranking error is semantic, not structural.")


if __name__ == "__main__":
    main()
