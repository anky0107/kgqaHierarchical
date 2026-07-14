"""
exp26_mpnet_hard_negatives.py
=============================

Generates adversarial hard negative candidates to train the Stage 2 MPNet ranker.
Instead of random negatives, this script mines paths that share high lexical overlap 
with the question but have incorrect structural semantics.

Usage:
  python train/exp26_mpnet_hard_negatives.py
"""

import os
import json
import argparse
from tqdm import tqdm

def calculate_overlap(text1, text2):
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))

def mine_hard_negatives(train_data_path, output_path, top_k_negs=15):
    print(f"Loading training data from {train_data_path}...")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Mining hard negatives based on lexical overlap...")
    
    hard_dataset = []
    
    for item in tqdm(data, desc="Mining"):
        q = item.get("question", "")
        cands = item.get("candidates", [])
        
        golds = [c for c in cands if c.get("is_gold")]
        negs = [c for c in cands if not c.get("is_gold")]
        
        if not golds or not negs:
            continue
            
        # Score negatives based on overlap with question
        for c in negs:
            name = c.get("name", "")
            path_str = str(c.get("path") or "")
            # We want paths or names that overlap heavily with the question
            c["_lex_score"] = calculate_overlap(q, name + " " + path_str.replace("->", " ").replace(".", " "))
            
        # Sort negatives by lexical overlap descending
        negs.sort(key=lambda x: x["_lex_score"], reverse=True)
        
        # Take the top K adversarial negatives
        hard_negs = negs[:top_k_negs]
        
        # Clean up
        for c in hard_negs:
            del c["_lex_score"]
            
        # Create new item with only golds and hard negs
        new_item = {
            "question": q,
            "path": item.get("path", ""),
            "candidates": golds + hard_negs
        }
        hard_dataset.append(new_item)

    print(f"Saving adversarial dataset ({len(hard_dataset)} samples) to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hard_dataset, f)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We use the full candidate set from exp16
    parser.add_argument("--train_data", type=str, default="data/exp16_cds_train_full.json")
    parser.add_argument("--output", type=str, default="data/exp26_s2_hard_negatives.json")
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()
    
    # Run from root dir for paths to work if run locally
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(ROOT)
    
    mine_hard_negatives(args.train_data, args.output, args.top_k)
