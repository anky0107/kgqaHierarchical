import json
import os

def check_cwq_mid_coverage(mids_path, mid2name_path):
    print(f"Loading CWQ MIDs from {mids_path}...")
    with open(mids_path, 'r') as f:
        cwq_mids = set(line.strip() for line in f)
    
    print(f"Total CWQ MIDs: {len(cwq_mids):,}")

    if not os.path.exists(mid2name_path):
        print(f"Error: {mid2name_path} not found.")
        return

    print(f"Loading Names from {mid2name_path}...")
    with open(mid2name_path, 'r', encoding='utf-8') as f:
        mid2name = json.load(f)
    
    name_mids = set(mid2name.keys())
    
    missing = cwq_mids - name_mids
    print("\nCoverage Results for CWQ Entities:")
    print(f"  CWQ MIDs with names: {len(cwq_mids & name_mids):,}")
    print(f"  CWQ MIDs missing names: {len(missing):,}")
    
    if len(missing) > 0:
        coverage = (len(cwq_mids & name_mids) / len(cwq_mids)) * 100
        print(f"  Coverage: {coverage:.2f}%")
        
        print("\nSample missing CWQ MIDs:")
        for mid in list(missing)[:10]:
            print(f"  {mid}")
    else:
        print("  All CWQ entities have names!")

if __name__ == "__main__":
    MIDS_PATH = 'data/cwq_mids.txt'
    MID2NAME_PATH = 'data/master_mid2name.json'
    check_cwq_mid_coverage(MIDS_PATH, MID2NAME_PATH)
