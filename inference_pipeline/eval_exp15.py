import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.exp15_optimized import Exp15Optimized
from utils.sparql_parser import find_reasoning_path

def evaluate_exp15(split="dev", limit=None, ckpt=None):
    print(f"\n[Eval] Starting Exp 15 Evaluation on {split} set...")
    
    # Initialize Pipeline
    pipeline = Exp15Optimized(strl_ckpt=ckpt)
    
    # Load Data
    data_path = os.path.join(ROOT, f'data/cwq_{split}.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        path = find_reasoning_path(item['sparql'])
        if not path: continue
        
        # Extract Topic MID
        te = path[0][0].replace("ns:", "")
        
        samples.append({
            'q': item['question'],
            'te': te,
            'gold': set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
        })
        if limit and len(samples) >= limit: break
        
    print(f"[Eval] Loaded {len(samples)} valid samples.")
    
    stats = {
        'hit1': 0,      # Stage 3 Top-1 is correct
        'hit_n': 0,     # Gold is in the Stage 2 set
        'total_ents': 0,
        'count': 0,
        'dead_ends': 0
    }
    
    for s in tqdm(samples):
        try:
            # 1. Run Stages 1 & 2 (Inference)
            # We'll modify run_inference to return both Stage 2 entities and Stage 3 results
            
            # For efficiency, we just use the existing run_inference and check the printed results?
            # No, let's capture the return values.
            # I'll modify exp15_optimized.py's run_inference to return (stage2_mids, stage3_results)
            pass
        except Exception as e:
            print(f"Error on sample: {e}")
            continue

    # Actually, let's just implement the loop here to avoid modifying the other script too much
    # or just use the return values if I modify it.
    
    # I will modify exp15_optimized.py to return (current_entities, results)
    
    results_list = []
    for s in tqdm(samples):
        # We need to manually run the steps here to get stats
        
        # Stage 1 & 2
        inputs = pipeline.tokenizer(s['q'], return_tensors="pt", padding=True, truncation=True).to(pipeline.device)
        fwd = pipeline.agent(inputs['input_ids'], inputs['attention_mask'])
        
        current_entities = {s['te']}
        for h in range(4):
            action = torch.argmax(fwd['action_logits'][0, h]).item()
            if action == 3: break
            hop_repr = fwd['hop_reprs'][0, h]
            beam_ids = pipeline.get_semantic_beam_with_filter(hop_repr, current_entities, action)
            beam_rels = [pipeline.id2rel[rid] for rid in beam_ids]
            
            next_entities = set()
            beam_rel_set = set(beam_rels)
            for mid in current_entities:
                for rel, tgt in pipeline.kg_get_neighbors(mid):
                    if rel in beam_rel_set:
                        next_entities.add(tgt)
            if not next_entities: break
            current_entities = next_entities
            
        # Stats for Stage 2
        stats['count'] += 1
        stats['total_ents'] += len(current_entities)
        if not current_entities: stats['dead_ends'] += 1
        
        is_hit_n = any(mid in s['gold'] for mid in current_entities)
        if is_hit_n: stats['hit_n'] += 1
        
        # Stage 3
        # Use the pipeline's cross-encoder logic
        candidates = []
        for mid in current_entities:
            name = pipeline.mid2name.get(mid, None)
            if name: candidates.append((mid, name))
        
        is_hit1 = False
        if candidates:
            candidates = candidates[:100]
            questions = [s['q']] * len(candidates)
            names = [c[1] for c in candidates]
            with torch.no_grad():
                enc = pipeline.selector_tokenizer(questions, names, padding=True, truncation=True, return_tensors='pt').to(pipeline.device)
                logits = pipeline.selector_model(**enc).logits.squeeze(-1)
                top_idx = torch.argmax(logits).item()
                best_mid = candidates[top_idx][0]
                if best_mid in s['gold']:
                    is_hit1 = True
                    stats['hit1'] += 1
        
        results_list.append({
            'q': s['q'],
            'hit1': is_hit1,
            'hit_n': is_hit_n,
            'num_ents': len(current_entities)
        })

    # Summary
    hit1_acc = (stats['hit1'] / stats['count']) * 100 if stats['count'] > 0 else 0
    hitn_acc = (stats['hit_n'] / stats['count']) * 100 if stats['count'] > 0 else 0
    avg_ents = stats['total_ents'] / stats['count'] if stats['count'] > 0 else 0
    
    print("\n" + "="*40)
    print(f"EXP 15 EVALUATION RESULTS ({split.upper()})")
    print("="*40)
    print(f"Hit@1 Accuracy:   {hit1_acc:.2f}%")
    print(f"Hit@N Accuracy:   {hitn_acc:.2f}% (Recall after Stage 2)")
    print(f"Avg Entities:     {avg_ents:.2f}")
    print(f"Dead Ends:        {stats['dead_ends']} / {stats['count']}")
    print(f"Total Samples:    {stats['count']}")
    print("="*40)

    # Save Results
    out_path = os.path.join(ROOT, f'inference_pipeline/results_exp15_{split}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'hit1': hit1_acc,
            'hit_n': hitn_acc,
            'avg_ents': avg_ents,
            'samples': stats['count']
        }, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=200) # Quick check on 200 samples
    parser.add_argument("--ckpt", type=str, default="checkpoints/exp15_strl_epoch_12.pt")
    args = parser.parse_args()
    
    evaluate_exp15(split=args.split, limit=args.limit, ckpt=args.ckpt)
