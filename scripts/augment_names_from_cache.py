import json
import os
import sys
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ingest_m2n_cache():
    cache_path = os.path.join(ROOT, 'data/cwq_ianyunshi/CWQ/m2n_cache.json')
    master_path = os.path.join(ROOT, 'data/master_mid2name.json')
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found at {cache_path}")
        return

    print(f"Loading master names from {master_path}...")
    master_names = {}
    if os.path.exists(master_path):
        with open(master_path, 'r', encoding='utf-8') as f:
            master_names = json.load(f)
    
    initial_count = len(master_names)
    print(f"Initial master names: {initial_count}")

    print(f"Loading cache from {cache_path} (this may take a while)...")
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    print(f"Merging {len(cache_data)} entries from cache...")
    added = 0
    updated = 0
    
    for mid, name in tqdm(cache_data.items()):
        # Clean mid if needed
        clean_mid = mid.replace('ns:', '')
        if clean_mid not in master_names:
            master_names[clean_mid] = name
            added += 1
        else:
            # Optionally update if names are different? 
            # Usually the cache is better or same.
            if master_names[clean_mid] != name:
                master_names[clean_mid] = name
                updated += 1

    print(f"Done. Added: {added}, Updated: {updated}")
    print(f"Total names in master: {len(master_names)}")

    print("Saving master_mid2name.json...")
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(master_names, f, indent=2, ensure_ascii=False)
    print("Success.")

if __name__ == '__main__':
    ingest_m2n_cache()
