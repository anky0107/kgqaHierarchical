"""
Master runner v2: executes experiments sequentially on GPU,
streams output live, skips already-completed experiments.
"""
import subprocess, sys, os, re, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = ROOT

EXPERIMENTS = [
    # ("Exp 0", "Flat BERT Baseline",           "train/exp0_flat_baseline.py",   "checkpoints/exp0_relation_flat_best.pt"),
    # ("Exp 1", "Domain-Restricted Search",      "train/exp1_domain_baseline.py", "checkpoints/exp1_domain_best.pt"),
    # ("Exp 2", "Contrastive Path Discrim.",      "train/exp2_cpd.py",             "checkpoints/exp2_cpd_best.pt"),
    # ("Exp 3", "Progressive Constraint Tight.",  "train/exp3_pct.py",             "checkpoints/exp3_pct_best.pt"),
    # ("Exp 4", "Cross-Hop Coherence Planning",   "train/exp4_chcp.py",            "checkpoints/exp4_chcp_best.pt"),
    # ("Exp 4-RL", "CHCP + RL Fine-tuning",       "train/exp4_rl.py",              "checkpoints/exp4_rl_epoch_49.pt"),
    # ("Exp 6", "Unified Adaptive-CHCP",          "train/exp6_unified.py",         "checkpoints/exp6_unified_best.pt"),
    ("Exp 7", "Scaled RoBERTa-Large",           "train/exp7_roberta.py",         "checkpoints/exp7_roberta_epoch_29.pt"),
]

def parse_metrics(output: str):
    """Extract last-reported dev metrics from stdout."""
    hit1 = hit3 = "-"
    
    # Try all known output formats
    for pattern in [
        r"Dev Hit@1\s*:\s*([\d.]+)",
        r"Dev Acc[:\s]+([\d.]+)",
        r"Dev Loss:.*?Acc:\s*([\d.]+)",
        r"Dev Rel Acc:\s*([\d.]+)",
        r"PPO Epoch \d+ complete\. Loss:\s*([\d.]+)",
    ]:
        matches = re.findall(pattern, output)
        if matches:
            hit1 = matches[-1]
            break
    
    for pattern in [
        r"Dev Hit@3\s*:\s*([\d.]+)",
        r"Dev Top3:\s*([\d.]+)",
    ]:
        matches = re.findall(pattern, output)
        if matches:
            hit3 = matches[-1]
            break
    
    return hit1, hit3

def run_experiment(tag, desc, script, ckpt):
    """Run a single experiment with live output streaming."""
    script_path = os.path.join(ROOT, script)
    
    # We always re-run to ensure convergence at higher epochs
    print(f"\n{'='*60}")
    print(f"  STARTING {tag}: {desc}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n", flush=True)
    
    t0 = time.time()
    output_lines = []
    
    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            output_lines.append(line)
        
        proc.wait(timeout=21600)  # 6 hour max per experiment
        elapsed = time.time() - t0
        
        full_output = "\n".join(output_lines)
        hit1, hit3 = parse_metrics(full_output)
        
        if proc.returncode != 0:
            return hit1, hit3, f"Error (code {proc.returncode}, {elapsed:.0f}s)"
        else:
            return hit1, hit3, f"Done ({elapsed:.0f}s)"
            
    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.time() - t0
        full_output = "\n".join(output_lines)
        hit1, hit3 = parse_metrics(full_output)
        return hit1, hit3, f"Timeout ({elapsed:.0f}s)"
    except Exception as e:
        return "-", "-", f"Error: {e}"

def main():
    results = []
    
    for tag, desc, script, ckpt in EXPERIMENTS:
        hit1, hit3, status = run_experiment(tag, desc, script, ckpt)
        results.append((tag, desc, hit1, hit3, status))
        print(f"\n  >> {tag} result: Hit@1={hit1}, Hit@3={hit3}, {status}\n")
    
    # Write results.md
    results_path = os.path.join(ROOT, "results.md")
    with open(results_path, "w") as f:
        f.write("# KGQA Research Experiment Results\n\n")
        f.write("| Experiment | Model Description | Dev Hit@1 | Dev Hit@3 | Status |\n")
        f.write("|---|---|---|---|---|\n")
        for tag, desc, h1, h3, st in results:
            f.write(f"| **{tag}** | {desc} | {h1} | {h3} | {st} |\n")
        f.write("\n---\n\n## Performance Notes\n\n")
        f.write("- **GPU**: RTX 5070 Laptop (SM 12.0 / Blackwell)\n")
        f.write("- **PyTorch**: 2.11.0+cu128 with Mixed Precision (AMP)\n")
        f.write("- **Dataset**: ComplexWebQuestions (CWQ) 1.1\n")
        f.write("- **Training**: AdamW lr=2e-5, early stopping patience=3\n")
    
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE — results written to results.md")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
