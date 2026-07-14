"""
Exp 37: Build DPO Dataset for T5 Reranker
=========================================

Reads the existing SFT dataset (exp30_t5_mc_train.json) and converts it 
into the format required by DPOTrainer:
{
    "prompt": "<full MC prompt>",
    "chosen": "<gold answer>",
    "rejected": "<hard negative answer>"
}
"""

import os, sys, json, random
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

def extract_candidates(prompt):
    names = []
    for line in prompt.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Check if line starts with a number and a dot, e.g. "1. "
        if line[0].isdigit() and '. ' in line:
            # Check if the prefix before '. ' is purely digits
            prefix = line.split('. ')[0]
            if prefix.isdigit():
                name_part = line.split('. ', 1)[1]
                if ' (Path:' in name_part:
                    name = name_part.split(' (Path:')[0]
                else:
                    name = name_part
                names.append(name.strip())
    return names

def main():
    in_path = os.path.join(ROOT, "data/exp30_t5_mc_train.json")
    out_path = os.path.join(ROOT, "data/exp37_t5_dpo_train.json")
    
    print(f"Loading SFT data from {in_path}...")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dpo_data = []
    
    for item in tqdm(data, desc="Building DPO Pairs"):
        prompt = item["prompt"]
        chosen = item["target"]
        
        cands = extract_candidates(prompt)
        
        # Filter out the chosen answer to get pure distractors
        distractors = [c for c in cands if c != chosen]
        
        if not distractors:
            # If there are no distractors (unlikely), skip
            continue
            
        # Randomly sample a distractor to be the rejected response
        # Since the candidates were pre-filtered by MPNet to be the hardest 50,
        # any distractor here is a high-quality "hard negative".
        rejected = random.choice(distractors)
        
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
    print(f"\nGenerated {len(dpo_data)} DPO triplets.")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
