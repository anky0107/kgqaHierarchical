"""
evaluate.py — Benchmark the CDS pipeline against exp16_cds_dev.json.

Usage
-----
    # Evaluate with default v2 Stage 3
    python -m cds_pipeline.evaluate

    # Evaluate with v3 (path-aware) Stage 3
    python -m cds_pipeline.evaluate --s3 v3

    # Compare v2 vs v3 in one run
    python -m cds_pipeline.evaluate --compare

    # Custom dev file
    python -m cds_pipeline.evaluate --dev path/to/cds_dev.json

Output
------
  Prints Hit@1 / Hit@3 / Hit@10 for the evaluated configuration(s).
  Saves results to metrics/cds_eval_<s3_version>.json
"""
from __future__ import annotations
import os, sys, json, argparse, time
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cds_pipeline.pipeline import CDSPipeline


# ─────────────────────────────────────────────────────────────
#  Core evaluator
# ─────────────────────────────────────────────────────────────

def run_eval(pipeline: CDSPipeline, dev_path: str) -> dict:
    """
    Evaluate pipeline on a CDS dev JSON file.
    Returns dict with hit@1/3/10, total, elapsed_s.
    """
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Only evaluate items that actually have a gold-labelled candidate
    samples = [s for s in data if any(c["is_gold"] for c in s["candidates"])]
    print(f"[Eval] {len(samples)} samples with gold labels  "
          f"(skipped {len(data) - len(samples)} without gold)")

    hits1 = hits3 = hits10 = 0
    total = 0
    t0    = time.time()

    for item in tqdm(samples, desc=f"CDS-{pipeline.s3_version}", ncols=80):
        q     = str(item["question"])
        cands = item["candidates"]
        path  = item.get("path")       # list[list[str]] — flatten_path handles it

        ranked = pipeline.rank(q, cands, path)
        total += 1
        if not ranked:
            continue

        # Build gold name set (some questions have multiple gold answers)
        gold_names = {c["name"] for c in cands if c["is_gold"]}

        for rank, c in enumerate(ranked[:10], start=1):
            if c["name"] in gold_names:
                if rank == 1:  hits1  += 1
                if rank <= 3:  hits3  += 1
                if rank <= 10: hits10 += 1
                break

    elapsed = time.time() - t0
    return {
        "s3_version": pipeline.s3_version,
        "total":      total,
        "hit@1":      round(hits1  / total * 100, 2),
        "hit@3":      round(hits3  / total * 100, 2),
        "hit@10":     round(hits10 / total * 100, 2),
        "elapsed_s":  round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    ver = results["s3_version"].upper()
    print()
    print("=" * 52)
    print(f"  CDS EVALUATION -- Stage 3 = {ver}")
    print("=" * 52)
    print(f"  Hit@1   :  {results['hit@1']:6.2f}%")
    print(f"  Hit@3   :  {results['hit@3']:6.2f}%")
    print(f"  Hit@10  :  {results['hit@10']:6.2f}%")
    print(f"  Samples :  {results['total']}")
    print(f"  Time    :  {results['elapsed_s']}s")
    print("=" * 52)


def save_results(results: dict) -> str:
    metrics_dir = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    out = os.path.join(metrics_dir, f"cds_eval_{results['s3_version']}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved -> {out}")
    return out


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the CDS pipeline on exp16_cds_dev.json"
    )
    parser.add_argument(
        "--s3", default="v2", choices=["v2", "v3"],
        help="Stage 3 checkpoint: v2=name-only, v3=path-aware (default: v2)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run BOTH v2 and v3 back-to-back for comparison",
    )
    parser.add_argument(
        "--dev", default=None,
        help="Path to CDS dev JSON (default: data/exp16_cds_dev.json)",
    )
    args = parser.parse_args()

    dev_path = args.dev or os.path.join(ROOT, "data", "exp16_cds_dev.json")
    if not os.path.exists(dev_path):
        sys.exit(f"[ERROR] Dev file not found: {dev_path}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    versions = ["v2", "v3"] if args.compare else [args.s3]

    all_results = []
    for ver in versions:
        print(f"\n" + "-" * 52)
        print(f"  Loading pipeline -- S3 version: {ver}")
        print("-" * 52)
        pipeline = CDSPipeline(device=device, s3_version=ver)
        res      = run_eval(pipeline, dev_path)
        print_results(res)
        save_results(res)
        all_results.append(res)
        del pipeline          # free VRAM before loading next version
        torch.cuda.empty_cache()

    if args.compare and len(all_results) == 2:
        v2, v3 = all_results
        delta  = v3["hit@1"] - v2["hit@1"]
        print(f"\n  Delta v3 - v2:  {delta:+.2f}%  Hit@1")


if __name__ == "__main__":
    main()
