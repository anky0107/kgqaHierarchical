# utils/verify.py

import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_models():
    print("========================================")
    print("KGQA CWQ Experiments Verification Script")
    print("========================================")
    
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        print("No checkpoints found. Please run training scripts first.")
        return
        
    exps = {
        "Exp 0 (Flat Baseline)": "exp0_relation_flat_best.pt",
        "Exp 1 (Domain Baseline)": "exp1_domain_best.pt",
        "Exp 2 (CPD)": "exp2_cpd_best.pt",
        "Exp 3 (PCT)": "exp3_pct_best.pt",
        "Exp 4 (CHCP)": "exp4_chcp_best.pt"
    }
    
    for name, ckpt in exps.items():
        path = os.path.join(ckpt_dir, ckpt)
        status = "LOADED" if os.path.exists(path) else "MISSING"
        print(f"{name:<30} | Status: {status}")
        
    print("\nTo run full verification, ensure testing datasets are loaded and run:")
    print("1. Hits@1, Hits@5, F1 computations for each model on the dev/test sets")
    print("2. Ablation generation (toggling coherence loss, confidence constraints, etc)")

if __name__ == "__main__":
    evaluate_models()
