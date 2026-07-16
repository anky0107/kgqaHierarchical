"""
run_full_pipeline.py -- Full End-to-End Training + Evaluation Orchestrator
===========================================================================

Runs the complete pipeline in order:

  STAGE A -- Hard Negative Mining (Exp 20)
      Generates exp18_cds_train_hard_full.json from the full 27k training set.
      (Skip if already done.)

  STAGE B -- Stage 3 Training on Full Hard Negatives (Exp 23)
      Trains BAAI/bge-reranker-base on the full 27k hard negatives.
      Saves checkpoints/exp23_s3_full_hard_neg.pt

  STAGE C -- Beam Search Evaluation (Exp 21) with best available S3
      Runs evaluate_stage3_cds.py for all available S3 versions.

  STAGE D -- Ensemble Evaluation (Exp 22) with best available S3
      (Deprecated in new pipeline, evaluate_stage3_cds.py handles end-to-end)

Usage:
    python run_full_pipeline.py                 # Full pipeline
    python run_full_pipeline.py --skip_mining   # Skip hard negative mining (Exp 20 already done)
    python run_full_pipeline.py --skip_training # Skip training (Exp 23 checkpoint exists)
    python run_full_pipeline.py --eval_only     # Only run evaluations

Results are saved to metrics/pipeline_results.json
"""

import os, sys, subprocess, json, time, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(ROOT, "metrics")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
os.makedirs(METRICS, exist_ok=True)

RESULTS_FILE = os.path.join(METRICS, "pipeline_results.json")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, desc):
    log(f"STARTING: {desc}")
    log(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        log(f"  ERROR: {desc} failed with return code {ret.returncode}")
        return False, elapsed
    log(f"  DONE: {desc}  ({elapsed/60:.1f} min)")
    return True, elapsed


def record_result(results, key, value):
    results[key] = value
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_mining",   action="store_true",
                        help="Skip Exp 20 hard negative mining (file already exists)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip Exp 23 training (checkpoint already exists)")
    parser.add_argument("--eval_only",     action="store_true",
                        help="Only run evaluations; skip mining and training")
    parser.add_argument("--resume_epoch",  type=int, default=None,
                        help="Resume training from a specific epoch checkpoint")
    args = parser.parse_args()

    if args.eval_only:
        args.skip_mining   = True
        args.skip_training = True

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    log("=" * 70)
    log("  FULL PIPELINE: Exp20 -> Exp23 -> Exp21 eval -> Exp22 eval")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE A: Hard Negative Mining (Exp 20)
    # -------------------------------------------------------------------------
    hard_full_path = os.path.join(ROOT, "data/exp18_cds_train_hard_full.json")

    if not args.skip_mining and not os.path.exists(hard_full_path):
        log("-" * 60)
        log("STAGE A: Exp 20 -- Hard Negative Mining (full 27k)")
        log("-" * 60)
        ok, elapsed = run(
            [sys.executable, "train/exp18_hard_negative_mining.py", "--resume"],
            "Exp 20 Hard Negative Mining"
        )
        record_result(results, "exp20_mining", {
            "status": "ok" if ok else "failed",
            "elapsed_min": round(elapsed / 60, 1),
            "output_file": hard_full_path,
        })
        if not ok:
            log("ABORT: Hard negative mining failed. Cannot train Exp 23.")
            sys.exit(1)
    else:
        if os.path.exists(hard_full_path):
            log(f"STAGE A: SKIPPED -- {hard_full_path} already exists.")
        else:
            log("STAGE A: SKIPPED by flag (--skip_mining).")

    # -------------------------------------------------------------------------
    # STAGE B: Train Stage 3 on Full Hard Negatives (Exp 23)
    # -------------------------------------------------------------------------
    exp23_ckpt = os.path.join(CHECKPOINTS, "exp23_s3_full_hard_neg.pt")

    if not args.skip_training and not os.path.exists(exp23_ckpt):
        log("-" * 60)
        log("STAGE B: Exp 23 -- Stage 3 Training (full 27k hard negatives)")
        log("-" * 60)
        if not os.path.exists(hard_full_path):
            log(f"ERROR: Training data not found: {hard_full_path}")
            log("       Run stage A first or use --skip_training.")
            sys.exit(1)
        
        cmd = [sys.executable, "train/exp23_s3_full_hard_neg.py"]
        if args.resume_epoch is not None:
            cmd.extend(["--resume_epoch", str(args.resume_epoch)])
            
        ok, elapsed = run(
            cmd,
            "Exp 23 Stage 3 Training"
        )
        record_result(results, "exp23_training", {
            "status": "ok" if ok else "failed",
            "elapsed_min": round(elapsed / 60, 1),
            "checkpoint": exp23_ckpt,
        })
        if not ok:
            log("WARNING: Exp 23 training failed. Will still run evals with v5.")
    else:
        if os.path.exists(exp23_ckpt):
            log(f"STAGE B: SKIPPED -- checkpoint already exists: {exp23_ckpt}")
        else:
            log("STAGE B: SKIPPED by flag (--skip_training).")

    # Determine best available S3 version
    best_s3 = "v6" if os.path.exists(exp23_ckpt) else "v5"
    log(f"Best available S3 version: {best_s3}")

    # Determine which S3 versions to evaluate
    s3_versions = ["v5"]
    if os.path.exists(exp23_ckpt):
        s3_versions.append("v6")

    # -------------------------------------------------------------------------
    # STAGE C: Beam Search Evaluation (Exp 21)
    # -------------------------------------------------------------------------
    log("-" * 60)
    log("STAGE C: Exp 21 -- CDS Entity Evaluation")
    log("-" * 60)

    # We now use our new script that actually connects F1 -> F2 -> F3!
    ok, elapsed = run(
        [sys.executable, "eval/evaluate_stage3_cds.py"],
        f"Eval Entity-Level CDS"
    )
    record_result(results, f"eval_cds", {
        "status": "ok" if ok else "failed",
        "elapsed_min": round(elapsed / 60, 1),
    })

    log("=" * 70)
    log("  PIPELINE COMPLETE")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("=" * 70)
    log("  PIPELINE COMPLETE -- Results summary:")
    log("=" * 70)
    for k, v in results.items():
        status = v.get("status", "?")
        elapsed = v.get("elapsed_min", 0)
        log(f"  {k:45s}  status={status}  time={elapsed:.1f}min")
    log(f"\n  Full results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
