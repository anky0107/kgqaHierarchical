"""
run_full_eval.py
================
Orchestrator to run the full dev-set evaluation for the v11_gen_sc model.
This script will evaluate all 3500 dev questions and log the metrics.
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Starting full evaluation of v11_gen_sc on the Dev Set...")
    cmd = [
        sys.executable, "-m", "cds_pipeline.evaluate_e2e",
        "--agent", "exp15",
        "--s3", "v11_gen_sc"
        # No max_samples limits
    ]
    subprocess.run(cmd, cwd=ROOT)
    print("Full evaluation complete.")

if __name__ == "__main__":
    main()
