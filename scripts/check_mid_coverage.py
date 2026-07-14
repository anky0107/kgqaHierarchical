import torch
import json
import os
import time

def check_mid_coverage(kg_path, mid2name_path):
    start = time.time()
    print(f"Loading KG from {kg_path}...")
    kg = torch.load(kg_path, map_location='cpu')
    
    print(f"Extracting MIDs from KG... (Triples: {sum(len(v) for v in kg['forward'].values()):,})")
    # Using set comprehension for speed
    subj_mids = set(kg['forward'].keys())
    obj_mids = set(kg['backward'].keys())
    
    # Extract all targets from forward edges efficiently
    targets = {tgt for neighbors in kg['forward'].values() for rel, tgt in neighbors}
    sources = {src for neighbors in kg['backward'].values() for rel, src in neighbors}
    
    kg_mids = subj_mids | obj_mids | targets | sources
    
    print(f"Total Unique MIDs in KG: {len(kg_mids):,}")

    if not os.path.exists(mid2name_path):
        print(f"Error: {mid2name_path} not found.")
        return

    print(f"Loading Names from {mid2name_path}...")
    with open(mid2name_path, 'r', encoding='utf-8') as f:
        mid2name = json.load(f)
    
    name_mids = set(mid2name.keys())
    print(f"Total Unique MIDs in mid2name: {len(name_mids):,}")
    
    missing = kg_mids - name_mids
    print("\nCoverage Results:")
    print(f"  MIDs with names: {len(kg_mids & name_mids):,}")
    print(f"  MIDs missing names: {len(missing):,}")
    
    if len(missing) > 0:
        coverage = (len(kg_mids & name_mids) / len(kg_mids)) * 100
        print(f"  Overall Coverage: {coverage:.2f}%")
        
        # Save missing MIDs to a file
        with open('data/missing_names_mids.txt', 'w') as f:
            for mid in sorted(missing):
                f.write(mid + '\n')
        print(f"\nMissing MIDs saved to data/missing_names_mids.txt")
    
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    KG_PATH = 'data/processed_kg/augmented_kg.pt'
    MID2NAME_PATH = 'data/master_mid2name.json'
    check_mid_coverage(KG_PATH, MID2NAME_PATH)
