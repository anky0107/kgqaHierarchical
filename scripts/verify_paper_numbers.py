"""
Verify paper-facing KGQA numbers from the actual checkpoints and pipeline.

This script evaluates Exp7, Exp9, and/or Exp15 on a CWQ split and reports:
  - raw answer recall: gold answer appears in the traversed candidate set
  - final Hit@1: CDS-selected entity is gold, if --cds is enabled
  - average candidate count after traversal
  - average hops executed
  - average relation fanout selected
  - latency and CUDA peak memory

Examples:
  python scripts/verify_paper_numbers.py --split dev --models exp7 exp9 --cds none
  python scripts/verify_paper_numbers.py --split dev --models exp9 exp15 --cds v3
  python scripts/verify_paper_numbers.py --split dev --limit 100 --models exp15 --cds v3
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime

import lmdb
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, RobertaTokenizer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from utils.sparql_parser import find_reasoning_path


DEFAULT_CKPTS = {
    "exp7": os.path.join(ROOT, "checkpoints", "exp7_roberta_best.pt"),
    "exp9": os.path.join(ROOT, "checkpoints", "exp9_rlmc_epoch_9.pt"),
    "exp15": os.path.join(ROOT, "checkpoints", "exp15_strl_epoch_19.pt"),
}


def discover_checkpoint(model_name, requested_path):
    if requested_path and os.path.exists(requested_path):
        return requested_path

    patterns = {
        "exp7": ["exp7_roberta_best.pt", "exp7_roberta_epoch_*.pt"],
        "exp9": ["exp9_rlmc_best.pt", "exp9_rlmc_epoch_*.pt"],
        "exp15": ["exp15_strl_epoch_19.pt", "exp15_strl_best.pt", "exp15_strl_epoch_*.pt"],
    }
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    import glob

    matches = []
    for pattern in patterns.get(model_name, []):
        matches.extend(glob.glob(os.path.join(ckpt_dir, pattern)))

    if not matches:
        return None

    def sort_key(path):
        base = os.path.basename(path)
        if "best" in base:
            return (1, 10**9, base)
        digits = [int(part) for part in base.replace(".", "_").split("_") if part.isdigit()]
        return (0, digits[-1] if digits else -1, base)

    return sorted(set(matches), key=sort_key)[-1]


def load_samples(split, limit=None):
    path = os.path.join(ROOT, "data", f"cwq_{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        path_info = find_reasoning_path(item.get("sparql", ""))
        if not path_info:
            continue
        topic = path_info[0][0].replace("ns:", "")
        gold = {
            ans.get("answer_id", "").replace("ns:", "")
            for ans in item.get("answers", [])
            if ans.get("answer_id")
        }
        if not topic or not gold:
            continue
        samples.append(
            {
                "id": item.get("ID") or item.get("id") or "",
                "question": item["question"],
                "topic_entity": topic,
                "gold": gold,
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples


class KGIndex:
    def __init__(self):
        lmdb_path = os.path.join(ROOT, "data", "processed_kg", "augmented_kg_lmdb")
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def lookup(self, entities, rels):
        next_entities = set()
        matched_edges = 0
        scanned_edges = 0
        rels_set = set(rels)
        with self.env.begin() as txn:
            for ent in entities:
                f_data = txn.get(f"f:{ent}".encode("utf-8"))
                if f_data:
                    for rel, tgt in pickle.loads(f_data):
                        scanned_edges += 1
                        if rel in rels_set:
                            next_entities.add(tgt)
                            matched_edges += 1

                b_data = txn.get(f"b:{ent}".encode("utf-8"))
                if b_data:
                    for rel, src in pickle.loads(b_data):
                        scanned_edges += 1
                        if rel in rels_set:
                            next_entities.add(src)
                            matched_edges += 1

        return next_entities, matched_edges, scanned_edges


class PathAwareRankerV2(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        q_emb = self.encoder(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p_emb = self.encoder(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e_emb = self.encoder(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q_emb, p_emb, e_emb], dim=-1)).squeeze(-1)


class CascadingDustSeparatorV2:
    """
    CDS v2 from exp16v2_* checkpoints:
      S1: MiniLM bi-encoder + precomputed entity embeddings
      S2: MPNet path-aware fusion ranker
      S3: BGE reranker over (question, entity), not path-aware
    """

    def __init__(self, device):
        self.device = device
        print("[CDS v2] Initializing Cascading Stack...")

        self.s1_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.s1_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.s1_model.load_state_dict(
            torch.load(os.path.join(ROOT, "checkpoints", "exp16v2_s1_bi.pt"), map_location=device)
        )
        self.s1_model.eval()

        print("[CDS v2] Loading entity embedding bank to RAM...")
        data = torch.load(os.path.join(ROOT, "data", "exp16_entity_embs.pt"), map_location="cpu")
        self.all_mids = data["mids"]
        self.mid2idx = {mid: i for i, mid in enumerate(self.all_mids)}
        self.all_embs = data["embs"]
        del data

        self.s2_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.s2_model = PathAwareRankerV2().to(device)
        self.s2_model.load_state_dict(
            torch.load(os.path.join(ROOT, "checkpoints", "exp16v2_s2_path.pt"), map_location=device)
        )
        self.s2_model.eval()

        self.s3_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.s3_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
        self.s3_model.load_state_dict(
            torch.load(os.path.join(ROOT, "checkpoints", "exp16v2_s3_cross.pt"), map_location=device)
        )
        self.s3_model.eval()

        with open(os.path.join(ROOT, "data", "master_mid2name.json"), "r", encoding="utf-8") as f:
            self.mid2name = json.load(f)

    @torch.no_grad()
    def separate_dust(self, question, path_str, candidate_mids):
        if not candidate_mids:
            return None

        q_enc = self.s1_tok(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        q_emb = self.s1_model(**q_enc).last_hidden_state[:, 0, :].cpu()

        found_mids = [mid for mid in candidate_mids if mid in self.mid2idx]
        if not found_mids:
            return list(candidate_mids)[0]

        e_embs = self.all_embs[[self.mid2idx[mid] for mid in found_mids]]
        sims = F.cosine_similarity(q_emb, e_embs)
        top_idx1 = torch.topk(sims, min(100, len(found_mids))).indices.tolist()
        mids1 = [found_mids[i] for i in top_idx1]
        names1 = [self.mid2name.get(mid, "Unknown") for mid in mids1]

        q2 = self.s2_tok([question] * len(mids1), return_tensors="pt", padding=True, truncation=True).to(self.device)
        p2 = self.s2_tok([path_str] * len(mids1), return_tensors="pt", padding=True, truncation=True).to(self.device)
        e2 = self.s2_tok(names1, return_tensors="pt", padding=True, truncation=True).to(self.device)
        scores2 = self.s2_model(
            q2["input_ids"],
            q2["attention_mask"],
            p2["input_ids"],
            p2["attention_mask"],
            e2["input_ids"],
            e2["attention_mask"],
        )

        top_idx2 = torch.topk(scores2, min(20, len(mids1))).indices.tolist()
        mids2 = [mids1[i] for i in top_idx2]
        names2 = [names1[i] for i in top_idx2]

        # v2 final judge sees only question/entity. v3 is the path-aware variant.
        enc3 = self.s3_tok(
            [question] * len(mids2),
            names2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        logits3 = self.s3_model(**enc3).logits.squeeze(-1)
        return mids2[torch.argmax(logits3).item()]


class Verifier:
    def __init__(self, device, cds_mode):
        self.device = device
        data_dir = os.path.join(ROOT, "data", "processed_entity")
        self.rel2id = torch.load(os.path.join(data_dir, "relation2id.pt"), map_location="cpu")
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, "domain2id.pt"), map_location="cpu")
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.kg = KGIndex()
        self.rel_emb_bank = None
        self.cds = self._load_cds(cds_mode)

    def _load_cds(self, cds_mode):
        if cds_mode == "none":
            return None
        if cds_mode == "v1":
            from inference_pipeline.benchmark_final_cds import CascadingDustSeparator

            return CascadingDustSeparator(self.device)
        if cds_mode == "v2":
            return CascadingDustSeparatorV2(self.device)
        if cds_mode == "v3":
            from inference_pipeline.benchmark_final_cds_v3 import CascadingDustSeparatorV3

            return CascadingDustSeparatorV3(self.device)
        raise ValueError(f"Unknown CDS mode: {cds_mode}")

    def load_model(self, model_name, ckpt_path):
        if model_name == "exp7":
            model = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        elif model_name == "exp9":
            base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
            model = RLConstraintAgent(base).to(self.device)
        elif model_name == "exp15":
            base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
            model = STRLAgent(base).to(self.device)
            if self.rel_emb_bank is None:
                self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
                self.rel_emb_bank.eval()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()
        return model

    @torch.no_grad()
    def traverse(self, model_name, model, question, topic_entity):
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        current = {topic_entity}
        path_log = []
        hops = 0
        selected_relation_counts = []
        matched_edges_total = 0
        scanned_edges_total = 0

        if model_name == "exp7":
            out = model(inputs["input_ids"], inputs["attention_mask"])
            for h in range(4):
                if torch.sigmoid(out["stop_logits"][0, h]).item() < 0.5:
                    break
                rel = self.id2rel[torch.argmax(out["rel_logits"][0, h]).item()]
                active = [rel]
                current, matched, scanned = self.kg.lookup(current, active)
                path_log.append(rel)
                selected_relation_counts.append(len(active))
                matched_edges_total += matched
                scanned_edges_total += scanned
                hops += 1
                if not current:
                    break

        elif model_name == "exp9":
            action_logits, _, rel_logits, dom_logits = model(inputs["input_ids"], inputs["attention_mask"])
            pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
            domain_name = self.id2dom[pred_dom_id]
            for h in range(4):
                action = torch.argmax(action_logits[0, h]).item()
                if action == 3:
                    break
                if action == 0:
                    active = [self.id2rel[torch.argmax(rel_logits[0, h]).item()]]
                elif action == 1:
                    active = [self.id2rel[rid] for rid in torch.topk(rel_logits[0, h], 5).indices.tolist()]
                else:
                    active = [rel for rel in self.id2rel.values() if domain_name in rel]
                current, matched, scanned = self.kg.lookup(current, active)
                path_log.append(active[0] if active else "")
                selected_relation_counts.append(len(active))
                matched_edges_total += matched
                scanned_edges_total += scanned
                hops += 1
                if not current:
                    break

        elif model_name == "exp15":
            out = model(inputs["input_ids"], inputs["attention_mask"])
            all_rel_embs = self.rel_emb_bank.all()
            for h in range(4):
                action = torch.argmax(out["action_logits"][0, h]).item()
                if action == 3:
                    break
                sims = torch.mv(all_rel_embs, out["hop_reprs"][0, h])
                # This matches the existing benchmark_final_cds.py Exp15 behavior.
                k = {0: 5, 1: 10, 2: 50}.get(action, 5)
                top_ids = torch.topk(sims, k).indices.tolist()
                active = [self.id2rel[rid] for rid in top_ids]
                current, matched, scanned = self.kg.lookup(current, active)
                path_log.append(active[0] if active else "")
                selected_relation_counts.append(len(active))
                matched_edges_total += matched
                scanned_edges_total += scanned
                hops += 1
                if not current:
                    break

        avg_rel_fanout = (
            sum(selected_relation_counts) / len(selected_relation_counts)
            if selected_relation_counts
            else 0.0
        )
        return {
            "candidates": current,
            "path": " -> ".join(path_log),
            "hops": hops,
            "avg_selected_relations_per_hop": avg_rel_fanout,
            "matched_edges": matched_edges_total,
            "scanned_edges": scanned_edges_total,
        }

    def evaluate_model(self, model_name, ckpt_path, samples):
        model = self.load_model(model_name, ckpt_path)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        stats = {
            "count": 0,
            "raw_recall_hits": 0,
            "hit1_hits": 0,
            "total_candidates": 0,
            "total_hops": 0,
            "total_relation_fanout": 0.0,
            "total_matched_edges": 0,
            "total_scanned_edges": 0,
            "cds_evaluated": 0,
            "cds_gold_dropped_or_missed": 0,
        }
        examples = []
        start_time = time.perf_counter()

        for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
            traversal = self.traverse(
                model_name,
                model,
                sample["question"],
                sample["topic_entity"],
            )
            candidates = traversal["candidates"]
            raw_hit = any(mid in sample["gold"] for mid in candidates)

            pred = None
            hit1 = False
            if self.cds is not None and candidates:
                pred = self.cds.separate_dust(sample["question"], traversal["path"], candidates)
                hit1 = pred in sample["gold"]
                stats["cds_evaluated"] += 1
                if raw_hit and not hit1:
                    stats["cds_gold_dropped_or_missed"] += 1

            stats["count"] += 1
            stats["raw_recall_hits"] += int(raw_hit)
            stats["hit1_hits"] += int(hit1)
            stats["total_candidates"] += len(candidates)
            stats["total_hops"] += traversal["hops"]
            stats["total_relation_fanout"] += traversal["avg_selected_relations_per_hop"]
            stats["total_matched_edges"] += traversal["matched_edges"]
            stats["total_scanned_edges"] += traversal["scanned_edges"]

            if len(examples) < 10:
                examples.append(
                    {
                        "id": sample["id"],
                        "question": sample["question"],
                        "topic_entity": sample["topic_entity"],
                        "gold": sorted(sample["gold"]),
                        "path": traversal["path"],
                        "candidate_count": len(candidates),
                        "raw_hit": raw_hit,
                        "predicted": pred,
                        "hit1": hit1,
                    }
                )

        elapsed = time.perf_counter() - start_time
        count = max(stats["count"], 1)
        result = {
            "model": model_name,
            "checkpoint": ckpt_path,
            "samples": stats["count"],
            "raw_recall_percent": 100.0 * stats["raw_recall_hits"] / count,
            "hit1_percent": 100.0 * stats["hit1_hits"] / count if self.cds is not None else None,
            "avg_candidates": stats["total_candidates"] / count,
            "avg_hops": stats["total_hops"] / count,
            "avg_selected_relations_per_hop": stats["total_relation_fanout"] / count,
            "avg_matched_edges": stats["total_matched_edges"] / count,
            "avg_scanned_edges": stats["total_scanned_edges"] / count,
            "latency_ms_per_question": 1000.0 * elapsed / count,
            "cds_evaluated": stats["cds_evaluated"],
            "cds_gold_dropped_or_missed": stats["cds_gold_dropped_or_missed"],
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if self.device.type == "cuda"
                else None
            ),
            "examples": examples,
        }
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Verify paper KGQA metrics from code.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick checks.")
    parser.add_argument("--models", nargs="+", default=["exp7", "exp9", "exp15"], choices=["exp7", "exp9", "exp15"])
    parser.add_argument(
        "--cds",
        default="none",
        choices=["none", "v1", "v2", "v3"],
        help="Apply no CDS, old CDS, CDS v2, or CDS v3.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--exp7-ckpt", default=DEFAULT_CKPTS["exp7"])
    parser.add_argument("--exp9-ckpt", default=DEFAULT_CKPTS["exp9"])
    parser.add_argument("--exp15-ckpt", default=DEFAULT_CKPTS["exp15"])
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Defaults to metrics/paper_number_verification_<timestamp>.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    requested_ckpts = {
        "exp7": args.exp7_ckpt,
        "exp9": args.exp9_ckpt,
        "exp15": args.exp15_ckpt,
    }
    ckpts = {
        name: discover_checkpoint(name, path)
        for name, path in requested_ckpts.items()
    }

    samples = load_samples(args.split, args.limit)
    verifier = Verifier(device=device, cds_mode=args.cds)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": ROOT,
        "split": args.split,
        "limit": args.limit,
        "device": str(device),
        "cds": args.cds,
        "num_samples": len(samples),
        "results": [],
    }

    for model_name in args.models:
        if ckpts[model_name] is None:
            missing = {
                "model": model_name,
                "requested_checkpoint": requested_ckpts[model_name],
                "status": "skipped",
                "reason": "No compatible checkpoint found in checkpoints/.",
            }
            report["results"].append(missing)
            print(
                f"[SKIP] {model_name}: no compatible checkpoint found. "
                f"Requested {requested_ckpts[model_name]}"
            )
            continue
        report["results"].append(verifier.evaluate_model(model_name, ckpts[model_name], samples))

    out_path = args.out
    if not out_path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(ROOT, "metrics", f"paper_number_verification_{stamp}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nVerification summary")
    print("=" * 80)
    for result in report["results"]:
        if result.get("status") == "skipped":
            print(f"{result['model']}: skipped - {result['reason']}")
            continue
        hit1 = result["hit1_percent"]
        hit1_text = "n/a" if hit1 is None else f"{hit1:.2f}%"
        print(
            f"{result['model']}: "
            f"raw_recall={result['raw_recall_percent']:.2f}% | "
            f"hit1={hit1_text} | "
            f"avg_candidates={result['avg_candidates']:.2f} | "
            f"latency={result['latency_ms_per_question']:.2f} ms/q | "
            f"cuda_peak={result['cuda_peak_memory_mb']}"
        )
    print("=" * 80)
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
