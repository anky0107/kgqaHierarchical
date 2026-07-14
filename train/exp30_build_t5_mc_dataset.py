"""
Exp 30: Build T5 Multiple-Choice Dataset
========================================

Runs Stage 1 and Stage 2 to get the top 50 candidates (the absolute hardest negatives)
and formats them into the exact Multiple-Choice prompt that T5 will see during inference.
Saves a JSON file ready for Seq2Seq training.
"""

import os, sys, json, random, argparse
from tqdm import tqdm
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from cds_pipeline.pipeline import CDSPipeline
from cds_pipeline.utils import flatten_path, path_to_nl

def build_mc_prompt(question, candidates, global_path_str):
    prompt = f"Question: {question}\n\nCandidates:\n"
    gold_name = None
    for i, c in enumerate(candidates, 1):
        name = c.get("name", "").strip() or "[UNK]"
        if c.get("is_gold"):
            gold_name = name
            
        cand_path_str = c.get("path") or global_path_str or ""
        path_nl = path_to_nl(cand_path_str)
        if path_nl:
            prompt += f"{i}. {name} (Path: {path_nl})\n"
        else:
            prompt += f"{i}. {name}\n"
            
    prompt += "\nWhich of the above candidates is the correct answer to the question? Answer with the exact name."
    return prompt, gold_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    train_path = os.path.join(ROOT, "data/exp16_cds_train_full.json")
    out_path   = os.path.join(ROOT, "data/exp30_t5_mc_train.json")

    print(f"Loading raw candidates from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if args.max_samples:
        data = data[:args.max_samples]

    # Initialize pipeline
    # We use S2=50 to match exactly what happens at inference time
    pipeline = CDSPipeline(s1_top_k=200, s2_top_k=50)

    mc_dataset = []
    
    for item in tqdm(data, desc="Building T5 Prompts"):
        q = item["question"]
        cands = item["candidates"]

        # Fast skip if no golds
        if not any(c.get("is_gold") for c in cands):
            continue

        path_str = flatten_path(item.get("path"))

        # Cap large beams
        MAX_CANDS = 5000
        if len(cands) > MAX_CANDS:
            golds = [c for c in cands if c.get("is_gold")]
            negs  = [c for c in cands if not c.get("is_gold")]
            negs  = random.sample(negs, min(MAX_CANDS - len(golds), len(negs)))
            cands_for_pipeline = golds + negs
        else:
            cands_for_pipeline = cands

        # Run S1 and S2 to get the hardest 50
        with torch.no_grad():
            s1_cands = pipeline._stage1(q, cands_for_pipeline)
            s2_cands = pipeline._stage2(q, path_str, s1_cands)

        # Must have gold in top 50 to train effectively, otherwise it learns to hallucinate
        if not any(c.get("is_gold") for c in s2_cands):
            # We inject the gold answer forcefully if S2 dropped it
            gold_cand = next((c for c in cands_for_pipeline if c.get("is_gold")), None)
            if gold_cand:
                s2_cands = [gold_cand] + s2_cands[:49]
            else:
                continue
                
        # Shuffle the 50 candidates so the gold answer isn't always at the same index
        random.shuffle(s2_cands)

        # Build prompt
        prompt, gold_name = build_mc_prompt(q, s2_cands, path_str)
        
        if not gold_name:
            continue

        mc_dataset.append({
            "prompt": prompt,
            "target": gold_name,
            "question": q
        })

    print(f"\nSaving {len(mc_dataset)} MC samples to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mc_dataset, f)
    print("Done!")

if __name__ == "__main__":
    main()
