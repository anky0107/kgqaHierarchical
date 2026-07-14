"""
CDS Pipeline Progress Monitor
==============================
Run this in a separate terminal to get live updates on the harvest -> train -> benchmark pipeline.

Usage:
    python utils/progress_monitor.py
"""
import os, sys, time, subprocess, json, datetime

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = {
    "harvest":   ("exp16v2_harvest.py",   "data/exp16v2_cds_train.json"),
    "training":  ("exp16v2_train.py",     "checkpoints/exp16v2_s3_cross.pt"),
    "benchmark": ("benchmark_final_cds.py", None),
}

CHECKPOINTS = {
    "s1": "checkpoints/exp16v2_s1_bi.pt",
    "s2": "checkpoints/exp16v2_s2_path.pt",
    "s3": "checkpoints/exp16v2_s3_cross.pt",
}

def sep(char="-", width=60): return char * width

def ts(): return datetime.datetime.now().strftime("%H:%M:%S")

def file_exists(rel): return os.path.isfile(os.path.join(ROOT, rel))

def file_size_mb(rel):
    p = os.path.join(ROOT, rel)
    return os.path.getsize(p) / 1e6 if os.path.isfile(p) else 0

def count_samples(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p): return 0
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data)
    except: return 0

def get_running_scripts():
    """Returns list of python scripts currently running."""
    try:
        import psutil
        scripts = []
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.info['name'] == 'python.exe' and p.info['cmdline']:
                cmdline = " ".join(p.info['cmdline'])
                for key in STAGES:
                    if STAGES[key][0] in cmdline:
                        scripts.append(key)
        return scripts
    except:
        return []

def print_status():
    running = get_running_scripts()
    print(f"\n{sep()}")
    print(f"  CDS PIPELINE MONITOR  [{ts()}]")
    print(sep())

    # ── Harvest ──────────────────────────────────────────────────
    harvest_file = "data/exp16v2_cds_train.json"
    harvest_done = file_exists(harvest_file)
    harvest_n    = count_samples(harvest_file)
    harvest_mb   = file_size_mb(harvest_file)
    harvest_running = "harvest" in running

    if harvest_running:
        pct = harvest_n / 27639 * 100 if harvest_n else 0
        eta_min = int((27639 - harvest_n) / max(harvest_n, 1) * (time.time() - START) / 60)
        print(f"  [HARVEST]   RUNNING  {harvest_n:>6,} / 27,639  ({pct:.1f}%)  ETA ~{eta_min}min")
    elif harvest_done:
        print(f"  [HARVEST]   DONE     {harvest_n:>6,} samples  ({harvest_mb:.0f} MB)")
    else:
        print(f"  [HARVEST]   PENDING  (not started)")

    # ── Checkpoints ───────────────────────────────────────────────
    print(f"\n  Checkpoints:")
    for name, path in CHECKPOINTS.items():
        if file_exists(path):
            mb = file_size_mb(path)
            print(f"    [{name.upper()}]  SAVED    {os.path.basename(path)}  ({mb:.0f} MB)")
        else:
            print(f"    [{name.upper()}]  MISSING  (not trained yet)")

    # ── Training ──────────────────────────────────────────────────
    train_running = "training" in running
    s1_done = file_exists(CHECKPOINTS["s1"])
    s2_done = file_exists(CHECKPOINTS["s2"])
    s3_done = file_exists(CHECKPOINTS["s3"])

    print(f"\n  [TRAINING]  ", end="")
    if train_running:
        stage = "S1" if not s1_done else ("S2" if not s2_done else "S3")
        print(f"RUNNING  (currently training {stage})")
    elif s3_done:
        print(f"DONE     All 3 stages saved")
    elif harvest_done:
        print(f"PENDING  (harvest done, run exp16v2_train.py)")
    else:
        print(f"PENDING  (waiting for harvest)")

    # ── Benchmark ─────────────────────────────────────────────────
    bench_running = "benchmark" in running
    print(f"\n  [BENCHMARK] ", end="")
    if bench_running:
        print("RUNNING  (evaluating models...)")
    elif s3_done:
        print("READY    (all checkpoints available, run benchmark)")
    else:
        print("PENDING  (waiting for training)")

    # ── Overall status ─────────────────────────────────────────────
    print(f"\n{sep()}")
    if bench_running:
        print("  STATUS: BENCHMARKING — final results incoming!")
    elif s3_done and not bench_running:
        print("  STATUS: Training COMPLETE — launch benchmark now")
    elif train_running:
        print("  STATUS: Training in progress")
    elif harvest_running:
        print("  STATUS: Harvesting data — training will start automatically")
    else:
        print("  STATUS: All stages pending")
    print(sep())

if __name__ == "__main__":
    START = time.time()
    INTERVAL = 60  # seconds between updates

    print(f"\nCDS Progress Monitor started at {ts()}", flush=True)
    print(f"Refreshing every {INTERVAL} seconds. Press Ctrl+C to stop.\n", flush=True)

    while True:
        try:
            print_status()
            sys.stdout.flush()
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[Monitor] Stopped at {ts()}.")
            break
