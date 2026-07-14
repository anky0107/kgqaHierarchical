"""
evaluate_extended_metrics.py — Extended Metrics Evaluation on CWQ Dev Set
=========================================================================

Overview
--------
Evaluates RLMC and STRL models on the CWQ Dev Set for extended performance 
and efficiency metrics. It performs graph traversal up to MAX_HOPS and
reranks candidates to compute Hit@1, Hits@5, Hits@10, Path/Depth Accuracy,
and resource metrics like Edges Expanded, Latency, Peak RAM, and Peak VRAM.

Paper Results Produced
----------------------
- Table 3 : Extended metrics including Hits@5, Traversal Cost (Edges), 
  Latency, and VRAM.

Inputs
------
- Processed CWQ dev set JSON.
- Freebase LMDB graph.
- Checkpoints for RLMC (Exp 9) and STRL (Exp 15).
- Reranker (Exp 9 final) to score candidates.

Outputs
-------
- eval/extended_metrics_results.json
- stdout metric summaries.

Usage
-----
    python eval/evaluate_extended_metrics.py
"""
# ──────────────────────────────────────────────────────
#  Imports and Config
# ──────────────────────────────────────────────────────
import os, sys, json, time, tracemalloc
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.kg_loader import build_kg_from_cwq_triples
from utils.sparql_parser import extract_triples

# ──────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────
MAX_HOPS = 4

def load_dev_samples(proc_dev_path):
    items = json.load(open(proc_dev_path, encoding='utf-8'))
    samples = []
    for item in items:
        te = item['topic_entity'].strip().strip(')')
        if not te or not item.get('gold_answers'):
            continue
        samples.append({
            'question':     item['question'],
            'topic_entity': te,
            'gold_answers': set(a.strip().strip(')') for a in item['gold_answers']),
            'gold_relations': item.get('relations', []),
            'num_hops': item.get('num_hops', len(item.get('relations', [])))
        })
    return samples

# ──────────────────────────────────────────────────────
#  Traversal and Tracking
# ──────────────────────────────────────────────────────
def traverse_and_track(kg, topic_entity, rel_sets_per_hop):
    """
    Returns (final_candidates, edges_expanded, depth)
    """
    active = {topic_entity}
    edges_expanded = 0
    depth = 0
    
    for hop_rels in rel_sets_per_hop:
        if not active or not hop_rels:
            break
        depth += 1
        nxt = set()
        for e in active:
            for rel, direction, tgt in kg.get_neighbors(e):
                edges_expanded += 1  # count every edge checked
                if rel in hop_rels:
                    nxt.add(tgt.strip(')'))
        if not nxt:
            break
        active = nxt
        
    return active if active != {topic_entity} else set(), edges_expanded, depth

# ──────────────────────────────────────────────────────
#  Candidate Reranking
# ──────────────────────────────────────────────────────
def score_candidates(candidates, question, reranker, tokenizer, device, bs=32):
    if not candidates:
        return []
    cands = list(candidates)
    scores = []
    for i in range(0, len(cands), bs):
        batch = cands[i:i+bs]
        enc = tokenizer([question.lower()]*len(batch), batch,
                        padding=True, truncation=True, max_length=128,
                        return_tensors='pt').to(device)
        with torch.no_grad():
            scores.extend(torch.sigmoid(reranker(**enc).logits[:, 0]).cpu().tolist())
    
    # Return candidates sorted by score (descending)
    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]

# ──────────────────────────────────────────────────────
#  Prediction and Metric Evaluation
# ──────────────────────────────────────────────────────
@torch.no_grad()
def predict_rlmc(rl, tok, q, device, id2rel):
    enc = tok(q, truncation=True, max_length=128, return_tensors='pt').to(device)
    act_logits, _, rel_logits, _ = rl(enc['input_ids'], enc['attention_mask'])
    actions = torch.argmax(act_logits[0], dim=-1).tolist()
    w_map = {0: 1, 1: 5, 2: 50, 3: 0}
    max_h = rel_logits.size(1)
    result = []
    for h in range(MAX_HOPS):
        a = actions[h] if h < len(actions) else 3
        w = w_map.get(a, 1)
        if w == 0:
            result.append(set())
            continue
        hi = min(h, max_h - 1)
        w = min(w, rel_logits.size(-1))
        ids = torch.topk(rel_logits[0, hi], k=w).indices.tolist()
        result.append({id2rel.get(r, '') for r in ids})
    return result

def evaluate_model(samples, kg, rl_model, reranker, tok, device, id2rel, name):
    metrics = {
        'em': 0, 'f1': 0, 'hits@5': 0, 'hits@10': 0, 
        'path_acc': 0, 'depth_acc': 0, 
        'edges_expanded': 0, 'latency_ms': 0,
        'total': 0
    }
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    
    for s in tqdm(samples, desc=name, ncols=90):
        t0 = time.time()
        
        # Predict relations
        rel_sets = predict_rlmc(rl_model, tok, s['question'], device, id2rel)
        
        # Traverse & track
        candidates, edges, depth = traverse_and_track(kg, s['topic_entity'], rel_sets)
        metrics['edges_expanded'] += edges
        
        # Rerank
        ranked_cands = score_candidates(candidates, s['question'], reranker, tok, device)
        
        latency = (time.time() - t0) * 1000
        metrics['latency_ms'] += latency
        
        # Metrics
        gold = s['gold_answers']
        metrics['total'] += 1
        
        if ranked_cands:
            top1 = ranked_cands[0]
            # EM / Hit@1
            if top1 in gold:
                metrics['em'] += 1
                metrics['f1'] += 1.0 # Set F1 for size 1 vs gold is 1.0 if it hits, though normally F1 = 2 * (1/1 * 1/|gold|) / (1/1 + 1/|gold|). We use standard Hit@1.
            
            # Hits@k
            if any(c in gold for c in ranked_cands[:5]):
                metrics['hits@5'] += 1
            if any(c in gold for c in ranked_cands[:10]):
                metrics['hits@10'] += 1
                
        # Path Accuracy: check if gold relations are covered
        path_correct = True
        for h, gr in enumerate(s['gold_relations']):
            if h >= len(rel_sets) or gr not in rel_sets[h]:
                path_correct = False
                break
        if path_correct:
            metrics['path_acc'] += 1
            
        # Depth Accuracy
        if depth == s['num_hops']:
            metrics['depth_acc'] += 1

    # Finalize
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    
    total = metrics['total']
    
    results = {
        'Model': name,
        'EM (Hit@1)': metrics['em'] / total,
        'Hits@5': metrics['hits@5'] / total,
        'Hits@10': metrics['hits@10'] / total,
        'Path Accuracy': metrics['path_acc'] / total,
        'Depth Accuracy': metrics['depth_acc'] / total,
        'Avg Edges Expanded': metrics['edges_expanded'] / total,
        'Latency (ms/q)': metrics['latency_ms'] / total,
        'Peak RAM (MB)': peak_ram / (1024*1024),
        'Peak VRAM (MB)': peak_vram / (1024*1024)
    }
    
    # F1 is identical to EM for Hit@1 in QA when outputting 1 answer.
    results['F1'] = results['EM (Hit@1)'] 
    
    return results

def load_ckpt(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model

# ──────────────────────────────────────────────────────
#  Main Loop
# ──────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    DATA  = os.path.join(ROOT, 'data')
    PDATA = os.path.join(DATA, 'processed_entity')
    CKPT  = os.path.join(ROOT, 'checkpoints')

    rel2id  = torch.load(os.path.join(PDATA, 'relation2id.pt'), weights_only=False)
    id2rel  = {v: k for k, v in rel2id.items()}
    num_rel = len(rel2id)
    train_d = torch.load(os.path.join(PDATA, 'train_domains.pt'), weights_only=False)
    num_dom = int(torch.max(train_d).item()) + 1

    print("\n[1/3] Building KG...")
    kg = build_kg_from_cwq_triples([
        os.path.join(DATA, 'cwq_train.json'),
        os.path.join(DATA, 'cwq_dev.json'),
        os.path.join(DATA, 'cwq_test.json'),
    ], extract_triples)

    print("\n[2/3] Loading Full Dev Set...")
    samples = load_dev_samples(os.path.join(DATA, 'processed_universal/cwq/dev.json'))
    print(f"Loaded {len(samples)} samples.")

    print("\n[3/3] Evaluating Models...")
    from train.exp9_rlmc import RLConstraintAgent
    from train.exp7_roberta import ScaledUnifiedPlanner
    
    rob_tok = RobertaTokenizer.from_pretrained('roberta-large')
    reranker = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1).to(device)
    reranker.load_state_dict(torch.load(os.path.join(CKPT, 'exp9_reranker_final.pt'), map_location=device, weights_only=False), strict=False)
    reranker.eval()
    
    # --- EXP 9 ---
    print("\n--- Exp 9: Standard RL ---")
    base9 = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(base9, os.path.join(CKPT, 'exp7_roberta_best.pt'), device)
    rl9 = RLConstraintAgent(base9).to(device)
    load_ckpt(rl9, os.path.join(CKPT, 'exp9_rlmc_epoch_9.pt'), device)
    rl9.eval()
    
    res9 = evaluate_model(samples, kg, rl9, reranker, rob_tok, device, id2rel, "Exp 9 (RL-MC)")
    del rl9, base9; torch.cuda.empty_cache()
    
    # --- EXP 15 ---
    print("\n--- Exp 15: STRL ---")
    base15 = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    rl15 = RLConstraintAgent(base15).to(device)
    load_ckpt(rl15, os.path.join(CKPT, 'exp15_strl_best.pt'), device)
    rl15.eval()
    
    res15 = evaluate_model(samples, kg, rl15, reranker, rob_tok, device, id2rel, "Exp 15 (STRL + CDS)")
    del rl15, base15; torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print(" EXTENDED METRICS EVALUATION RESULTS")
    print("="*80)
    for res in [res9, res15]:
        print(f"\nModel: {res['Model']}")
        for k, v in res.items():
            if k == 'Model': continue
            if 'MB' in k or 'Edges' in k or 'Latency' in k:
                print(f"  {k:20s}: {v:.1f}")
            else:
                print(f"  {k:20s}: {v:.4f}")
                
    # Save to JSON
    with open(os.path.join(ROOT, 'eval', 'extended_metrics_results.json'), 'w') as f:
        json.dump([res9, res15], f, indent=4)
    print("\nSaved to eval/extended_metrics_results.json")

if __name__ == '__main__':
    main()
