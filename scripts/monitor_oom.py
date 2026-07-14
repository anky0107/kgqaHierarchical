import os, sys, time, subprocess

# Configuration
PID = 28008
SCRIPT_PATH = "train/exp15_strl.py"
LOG_PATH = "metrics/monitor_log.txt"

def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{time.ctime()}] {msg}\n")
    print(msg)

log(f"Starting monitor for PID {PID}...")

while True:
    # 1. Check if process is still running
    try:
        # On Windows, tasklist or specialized psutil
        res = subprocess.run(["tasklist", "/fi", f"PID eq {PID}"], capture_output=True, text=True)
        if str(PID) not in res.stdout:
            log(f"Process {PID} is gone!")
            break
    except Exception as e:
        log(f"Monitor error: {e}")
        break
    
    time.sleep(10)

# 2. If we reach here, the process died. Assume OOM or Crash.
log("Attempting fallback to Batch Size = 4...")

# 3. Modify the script back to BS=4
try:
    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace("batch_size=8,  shuffle=True", "batch_size=4,  shuffle=True")
    
    with open(SCRIPT_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
    log("Reverted batch_size to 4.")
except Exception as e:
    log(f"Failed to revert script: {e}")
    sys.exit(1)

# 4. Restart the training
log("Restarting training...")
env = os.environ.copy()
env["PYTHONPATH"] = "."
# We use Popen so the monitor itself can finish or continue? 
# Actually, I'll just start it and exit the monitor.
subprocess.Popen([sys.executable, "-u", SCRIPT_PATH], env=env)
log("Training restarted. Monitor exiting.")
