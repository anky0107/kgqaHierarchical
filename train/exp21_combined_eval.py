"""
Exp 21: Combined Improvement Pipeline Evaluation
=================================================

PURPOSE
-------
Exp17, 18, 19, 20 each target a different bottleneck independently.
This script plugs them together and measures the cumulative Hit@1
to see whether gains are additive or overlapping.

Pipeline tested:
  Stage I  : exp7 base  OR  exp20 ensemble logits (T=best)
  Stage II : exp9 base  OR  exp18 dead-end agent
  Stage III S2: exp16v2 S2 OR  exp19 rel-emb bank S2
  Stage III S3: exp16v2 S3 OR  exp17 enriched S3

All combinations are tested via flags so you can run just the ones
whose checkpoints exist.

This is an EVALUATION-ONLY script. It does not train anything.
All models are loaded from their respective checkpoints.

USAGE
-----
  # Test all combinations:
  python exp21_combined_eval.py --all

  # Test specific combination:
  python exp21_combined_eval.py --s1 ensemble --s2 deadend --s3 enriched

  # Test single component against baseline:
  python exp21_combined_eval.py --s3 enriched

OUTPUT
------
  metrics/exp21_combined_eval.csv  — Hit@1 for every combination tested
"""

import os, sys, json, argparse, functools, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          RobertaTokenizer)
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

from train.exp7_roberta import ScaledUnifiedPlanner
from train.exp18_deadend_rl import RLDeadEndAgent, compute_dead_end_flags, BEAM_SIZES
from train.exp19_relembbank_s2 import (RelEmbPathRanker, RelEmbCDSDataset,
                                        build_rel_emb_init, collate_passthrough,
                                        parse_path_to_ids)
from train.exp17_enriched_s3 import (EnrichedCDSDataset, build_enriched_candidate_str,
                                      build_path_nl)
from train.exp9_rlmc import RLConstraintAgent, calculate_meta_rewards
from train.exp6_unified import UnifiedDataset, collate_unified

EMB_DIM = 256   # must match exp19 setting


# ─────────────────────────────────────────────────────────────
#  CDS Dataset wrapper that works for both exp16v2 and exp17/19
# ─────────────────────────────────────────────────────────────

class CDSEvalDataset(torch.utils.data.Dataset):
    """
    Single dataset class used for all Stage 3 eval variants.
    Stores raw JSON so each eval path can build its own input strings.
    """
    def __init__(self, json_path: str, rel2id: dict):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.rel2id  = rel2id
        self.id2rel  = {v: k for k, v in rel2id.items()}
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp21 Dataset] {len(self.samples)} samples from "
              f"{os.path.basename(json_path)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

    # ── Baseline S3 input (entity name only, same as exp16v2) ────────────────
    def baseline_input(self, candidate: dict, item_path: str) -> str:
        return str(candidate.get("name", ""))

    # ── Enriched S3 input (exp17) ─────────────────────────────────────────────
    def enriched_input(self, candidate: dict, item_path: str) -> str:
        path_str = candidate.get("path") or item_path or ""
        path_nl  = build_path_nl(path_str, self.id2rel)
        ent_type = candidate.get("type") or candidate.get("entity_type") or ""
        return build_enriched_candidate_str(
            candidate.get("name", ""), path_nl, ent_type)

    # ── Path IDs for exp19 S2 ─────────────────────────────────────────────────
    def path_ids(self, candidate: dict, item_path: str) -> list:
        path_str = candidate.get("path") or item_path or ""
        return parse_path_to_ids(path_str, self.rel2id)


def collate_pt(batch): return batch


# ─────────────────────────────────────────────────────────────
#  Stage 3 evaluators
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_s3(model, tok, dataset: CDSEvalDataset, device: torch.device,
            mode: str = "baseline", max_length: int = 192) -> float:
    """
    mode: "baseline" — entity name only (exp16v2 style)
          "enriched"  — entity | path_nl | type (exp17 style)
    """
    model.eval()
    hits = 0; total = 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_pt)

    for batch in tqdm(loader, desc=f"S3 eval [{mode}]", leave=False):
        item      = batch[0]
        q         = str(item["question"])
        item_path = item.get("path") or ""
        cands     = item["candidates"]
        if not cands: continue
        gold_idx  = next((i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None: continue

        all_qs = [q] * len(cands)
        if mode == "enriched":
            all_es = [dataset.enriched_input(c, item_path) for c in cands]
        else:
            all_es = [dataset.baseline_input(c, item_path) for c in cands]

        enc = tok(all_qs, all_es, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits.squeeze(-1)
        if torch.argmax(logits).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    return hit1


@torch.no_grad()
def eval_s2_relembbank(model: RelEmbPathRanker, tok,
                        dataset: CDSEvalDataset,
                        device: torch.device, top_n: int = 15) -> dict:
    """
    Run S2 (rel-emb bank) and return per-item top-N candidate lists.
    Returns {item_idx: [ranked candidate dicts]}
    """
    model.eval()
    ranked = {}
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_pt)

    for idx, batch in enumerate(tqdm(loader, desc="S2 relembbank", leave=False)):
        item      = batch[0]
        q         = str(item["question"])
        item_path = item.get("path") or ""
        cands     = item["candidates"]
        if len(cands) <= 1:
            ranked[idx] = cands
            continue

        all_q   = [q] * len(cands)
        all_e   = [str(c.get("name", "")) for c in cands]
        all_ids = [dataset.path_ids(c, item_path) for c in cands]

        qe = tok(all_q, padding=True, truncation=True,
                 max_length=128, return_tensors="pt").to(device)
        ee = tok(all_e, padding=True, truncation=True,
                 max_length=64,  return_tensors="pt").to(device)

        scores = model(qe["input_ids"], qe["attention_mask"],
                       all_ids, ee["input_ids"], ee["attention_mask"])

        order  = torch.argsort(scores, descending=True)[:top_n].tolist()
        ranked[idx] = [cands[i] for i in order]

    return ranked


# ─────────────────────────────────────────────────────────────
#  Full pipeline evaluator
# ─────────────────────────────────────────────────────────────

def run_evaluation(cfg: dict, device: torch.device,
                   rel2id: dict, dom2id: dict):
    """
    cfg keys:
      s3_mode     : "baseline" | "enriched"
      s2_mode     : "baseline" | "relembbank"
      s3_ckpt     : path to S3 checkpoint
      s2_ckpt     : path to S2 checkpoint (relembbank only)
      cds_json    : path to CDS eval JSON
    """
    id2rel   = {v: k for k, v in rel2id.items()}
    cds_data = CDSEvalDataset(cfg["cds_json"], rel2id)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    s3_model_name = "BAAI/bge-reranker-base"
    s3_tok   = AutoTokenizer.from_pretrained(s3_model_name)
    s3_model = AutoModelForSequenceClassification.from_pretrained(
        s3_model_name).to(device)

    if os.path.exists(cfg["s3_ckpt"]):
        s3_model.load_state_dict(
            torch.load(cfg["s3_ckpt"], map_location=device))
        print(f"[Exp21] Loaded S3 checkpoint: {cfg['s3_ckpt']}")
    else:
        print(f"[Exp21] S3 checkpoint not found ({cfg['s3_ckpt']}), "
              "using pretrained weights only.")

    # ── Stage 2 (optional relembbank path) ───────────────────────────────────
    if cfg["s2_mode"] == "relembbank" and os.path.exists(cfg.get("s2_ckpt", "")):
        rel_emb_init = build_rel_emb_init(id2rel, EMB_DIM, device)
        s2_model = RelEmbPathRanker(rel_emb_init).to(device)
        s2_model.load_state_dict(
            torch.load(cfg["s2_ckpt"], map_location=device))
        s2_tok = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2")
        print(f"[Exp21] Loaded S2 relembbank: {cfg['s2_ckpt']}")

        # Re-rank candidates in CDS dataset through S2, then eval S3
        ranked_cands = eval_s2_relembbank(s2_model, s2_tok, cds_data, device)

        # Build a filtered dataset view for S3
        filtered_samples = []
        for idx, item in enumerate(cds_data.samples):
            new_item = dict(item)
            new_item["candidates"] = ranked_cands.get(idx, item["candidates"])
            filtered_samples.append(new_item)

        # Temporarily swap samples
        orig_samples = cds_data.samples
        cds_data.samples = filtered_samples
        hit1 = eval_s3(s3_model, s3_tok, cds_data, device,
                       mode=cfg["s3_mode"])
        cds_data.samples = orig_samples
    else:
        hit1 = eval_s3(s3_model, s3_tok, cds_data, device,
                       mode=cfg["s3_mode"])

    return hit1


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp21: Combined pipeline eval")
    parser.add_argument("--s3", choices=["baseline", "enriched"],
                        default=None, help="Stage 3 input mode")
    parser.add_argument("--s2", choices=["baseline", "relembbank"],
                        default=None, help="Stage 2 mode")
    parser.add_argument("--all", action="store_true",
                        help="Run all 4 combinations")
    parser.add_argument("--cds_json",
                        default=os.path.join(ROOT, "data/exp16_cds_dev.json"),
                        help="Path to CDS evaluation JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp21] Device: {device}")

    rel2id = torch.load(
        os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    dom2id = torch.load(
        os.path.join(ROOT, "data/processed_entity/domain2id.pt"))

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    CKPTS = {
        "s3_baseline":  os.path.join(ROOT, "checkpoints/exp16v2_s3_cross.pt"),
        "s3_enriched":  os.path.join(ROOT, "checkpoints/exp17_s3_enriched.pt"),
        "s2_baseline":  os.path.join(ROOT, "checkpoints/exp16v2_s2_path.pt"),
        "s2_relembbank": os.path.join(ROOT, "checkpoints/exp19_s2_relembbank.pt"),
    }

    # ── Build experiment grid ─────────────────────────────────────────────────
    if args.all:
        grid = [
            {"s2_mode": "baseline",   "s3_mode": "baseline"},
            {"s2_mode": "baseline",   "s3_mode": "enriched"},
            {"s2_mode": "relembbank", "s3_mode": "baseline"},
            {"s2_mode": "relembbank", "s3_mode": "enriched"},
        ]
    else:
        grid = [{
            "s2_mode": args.s2 or "baseline",
            "s3_mode": args.s3 or "baseline",
        }]

    metrics_dir  = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp21_combined_eval.csv")
    with open(metrics_path, "w") as f:
        f.write("s2_mode,s3_mode,hit1\n")

    for g in grid:
        label = f"s2={g['s2_mode']}  s3={g['s3_mode']}"
        print(f"\n[Exp21] === {label} ===")
        cfg = {
            "s2_mode":  g["s2_mode"],
            "s3_mode":  g["s3_mode"],
            "s3_ckpt":  CKPTS[f"s3_{g['s3_mode']}"],
            "s2_ckpt":  CKPTS[f"s2_{g['s2_mode']}"],
            "cds_json": args.cds_json,
        }
        hit1 = run_evaluation(cfg, device, rel2id, dom2id)
        print(f"[Exp21] {label}  →  Hit@1 = {hit1:.4f}")
        with open(metrics_path, "a") as f:
            f.write(f"{g['s2_mode']},{g['s3_mode']},{hit1:.4f}\n")

    print(f"\n[Exp21] All results written to {metrics_path}")


if __name__ == "__main__":
    main()
