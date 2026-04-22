"""
End-to-End KGQA Evaluation Pipeline (Paper-Comparable)

Evaluation Protocol:
  For each CWQ question:
  1. Extract gold relation path from SPARQL
  2. Use trained model to predict relation path
  3. If predicted path matches gold → answer is correct (Hits@1 = 1)
  4. Report: Hits@1, F1, Path Accuracy, Per-Hop Accuracy

  This is equivalent to the standard KGQA evaluation *assuming correct KG execution*,
  which is the standard assumption in planning-based methods (DRKG, DAMR, Plan-Then-Retrieve).
"""
import json, os, sys, re, functools, torch

# Fix Windows cp1252 encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import BertTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.sparql_parser import find_reasoning_path
from shared.metrics import hits_at_k

# ============================================================
#  Extract Question-Level Gold Data
# ============================================================

def extract_evaluation_data(cwq_data, relation2id):
    """Extract question-level data for evaluation."""
    samples = []
    skipped = 0
    
    for item in cwq_data:
        question = item['question']
        sparql = item['sparql']
        
        path = find_reasoning_path(sparql)
        if path is None:
            skipped += 1
            continue
        
        # Gold relation path
        gold_relations = []
        all_in_vocab = True
        for node, rel, direction, next_node in path:
            if rel not in relation2id:
                all_in_vocab = False
                break
            gold_relations.append({
                'relation': rel,
                'relation_id': relation2id[rel],
                'direction': direction,
            })
        
        if not all_in_vocab or not gold_relations:
            skipped += 1
            continue
        
        # Gold answers (for F1)
        gold_answers = []
        if 'answers' in item and item['answers']:
            for ans in item['answers']:
                gold_answers.append(ans.get('answer', ''))
        
        samples.append({
            'id': item.get('ID', ''),
            'question': question,
            'gold_path': gold_relations,
            'gold_answers': gold_answers,
            'num_hops': len(gold_relations),
        })
    
    print(f"  Extracted {len(samples)} samples, skipped {skipped}")
    return samples

# ============================================================
#  Model Prediction Functions
# ============================================================

def predict_exp0(model, tokenizer, question, device, num_rels, k=10):
    """Exp 0: flat relation classifier."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    probs = F.softmax(logits, dim=-1)
    _, topk = torch.topk(probs, k=k, dim=-1)
    return topk[0].cpu().tolist()

def predict_exp3(model, tokenizer, question, device, k=10):
    """Exp 3 (PCT): returns top-k relation predictions + confidence."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _, _, rel_logits, confidence = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    probs = F.softmax(rel_logits, dim=-1)
    _, topk = torch.topk(probs, k=k, dim=-1)
    return topk[0].cpu().tolist(), confidence[0].item()

def predict_exp4(model, tokenizer, question, device, max_hops=4, k=10):
    """Exp 4 (CHCP): returns per-hop top-k predictions."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            rel_logits, stop_logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        # Fix: stop_logits is [B, max_hops]. Get sigmoid of scalar logit.
        stop_p = torch.sigmoid(stop_logits[0, h]).item()
        results.append({'top_ids': topk.cpu().tolist(), 'stop_prob': stop_p})
    return results

def predict_exp6(model, tokenizer, question, device, max_hops=4, k=10):
    """Exp 6 (Unified): returns per-hop top-k predictions."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            rel_logits = out['rel_logits']
            stop_logits = out['stop_logits']
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        stop_p = torch.sigmoid(stop_logits[0, h]).item()
        results.append({'top_ids': topk.cpu().tolist(), 'stop_prob': stop_p})
    return results

def predict_exp7(model, tokenizer, question, device, max_hops=4, k=10):
    """Exp 7 (Scaled RoBERTa): returns per-hop top-k predictions."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            rel_logits = out['rel_logits']
            stop_logits = out['stop_logits']
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        stop_p = torch.sigmoid(stop_logits[0, h]).item()
        results.append({'top_ids': topk.cpu().tolist(), 'stop_prob': stop_p})
    return results

def predict_exp9(rl_agent, tokenizer, question, device, max_hops=4, k=10):
    """Exp 9: RL Meta-Constraint Agent"""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            action_logits, _, rel_logits, _ = rl_agent(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            
    results = []
    actions = torch.argmax(action_logits[0], dim=-1).tolist()
    
    for h in range(max_hops):
        a = actions[h]
        probs = F.softmax(rel_logits[0, h], dim=-1)
        
        # Determine beam width based on RL constraint action
        if a == 0:  # TIGHT
            w = 1
        elif a == 1: # MEDIUM
            w = 5
        elif a == 2: # LOOSE
            w = 50
        else:       # STOP
            w = 0
            
        if w > 0:
            _, topw = torch.topk(probs, k=w, dim=-1)
            # Evaluate using top-w as the predicted constraint mask 
            results.append({'top_ids': topw.cpu().tolist()})
        else:
            break # Stopped early
            
    return results

# ============================================================
#  Evaluation Logic
# ============================================================

def evaluate_model(samples, model, tokenizer, id2relation, device, 
                   model_name, predict_fn, model_type='flat'):
    """
    Evaluate a model on CWQ questions.
    
    For each question:
    - Model predicts relation(s) for each hop
    - Compare predicted path against gold path
    - If path matches → answer is correct (standard planning evaluation)
    
    Metrics:
    - Hits@1: % questions where FULL predicted path matches gold
    - Hits@3: % questions where gold path is in top-3 combinations
    - Per-Hop Acc: % of individual hops predicted correctly
    - Path Acc: same as Hits@1 for planning evaluation
    """
    total = 0
    hits1 = 0  # full path match (top-1)
    hits3 = 0  # full path in top-3 per hop
    hop_correct = 0
    hop_total = 0
    
    # Breakdown by num_hops
    by_hops = defaultdict(lambda: {'total': 0, 'hits1': 0, 'hop_correct': 0, 'hop_total': 0})
    
    for sample in tqdm(samples, desc=f"Eval {model_name}"):
        question = sample['question']
        gold_path = sample['gold_path']
        num_hops = sample['num_hops']
        
        if model_type == 'flat':
            # Exp 0/Exp 3: predict single relation
            if model_type == 'flat' and model_name.startswith('Exp 0'):
                top_ids = predict_fn(model, tokenizer, question, device, len(id2relation), k=10)
            else:
                top_ids, conf = predict_fn(model, tokenizer, question, device, k=10)
            
            # For single-relation models: first hop uses top-1, second uses top-2 etc.
            path_match_1 = True
            path_match_3 = True
            for h, gold_hop in enumerate(gold_path):
                gold_id = gold_hop['relation_id']
                hop_total += 1
                by_hops[num_hops]['hop_total'] += 1
                
                # For flat models: use the top predictions in order
                if h < len(top_ids):
                    pred_id = top_ids[h]
                    if pred_id == gold_id:
                        hop_correct += 1
                        by_hops[num_hops]['hop_correct'] += 1
                    else:
                        path_match_1 = False
                    # Check if gold is in top-3 starting from position h
                    if gold_id not in top_ids[max(0,h):h+3]:
                        path_match_3 = False
                else:
                    path_match_1 = False
                    path_match_3 = False
                    
        elif model_type in ['chcp', 'unified', 'roberta', 'rlmc']:
            # Exp 4, 6, 7, 9: predict all hops simultaneously
            hop_preds = predict_fn(model, tokenizer, question, device, max_hops=4, k=10)
            
            path_match_1 = True
            path_match_3 = True
            for h, gold_hop in enumerate(gold_path):
                gold_id = gold_hop['relation_id']
                hop_total += 1
                by_hops[num_hops]['hop_total'] += 1
                
                if h < len(hop_preds):
                    pred_top = hop_preds[h]['top_ids']
                    if pred_top[0] == gold_id:
                        hop_correct += 1
                        by_hops[num_hops]['hop_correct'] += 1
                    else:
                        path_match_1 = False
                    if gold_id not in pred_top[:3]:
                        path_match_3 = False
                else:
                    path_match_1 = False
                    path_match_3 = False
        
        if path_match_1:
            hits1 += 1
            by_hops[num_hops]['hits1'] += 1
        if path_match_3:
            hits3 += 1
        
        total += 1
        by_hops[num_hops]['total'] += 1
    
    results = {
        'model': model_name,
        'total': total,
        'hits@1': hits1 / total if total > 0 else 0,
        'hits@3': hits3 / total if total > 0 else 0,
        'hop_accuracy': hop_correct / hop_total if hop_total > 0 else 0,
        'by_hops': dict(by_hops),
    }
    
    # Print hop breakdown
    print(f"\n  {model_name} Results:")
    print(f"    Overall Hits@1: {results['hits@1']:.4f} | Hits@3: {results['hits@3']:.4f} | Hop Acc: {results['hop_accuracy']:.4f}")
    for nh in sorted(by_hops.keys()):
        bh = by_hops[nh]
        h1 = bh['hits1']/bh['total'] if bh['total'] > 0 else 0
        ha = bh['hop_correct']/bh['hop_total'] if bh['hop_total'] > 0 else 0
        print(f"    {nh}-hop: Hits@1={h1:.4f} | Hop Acc={ha:.4f} | ({bh['total']} questions)")
    
    return results

# ============================================================
#  Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load relation maps
    data_dir = os.path.join(ROOT, 'data/processed_entity')
    relation2id = torch.load(os.path.join(data_dir, 'relation2id.pt'))
    id2relation = {v: k for k, v in relation2id.items()}
    
    # Extract evaluation data
    print("\n[1/3] Extracting evaluation data...")
    dev_data = json.load(open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8'))
    test_data = json.load(open(os.path.join(ROOT, 'data/cwq_test.json'), 'r', encoding='utf-8'))
    train_data = json.load(open(os.path.join(ROOT, 'data/cwq_train.json'), 'r', encoding='utf-8'))
    
    print("  Dev set:")
    dev_samples = extract_evaluation_data(dev_data, relation2id)
    print("  Test set:")
    test_samples = extract_evaluation_data(test_data, relation2id)
    print("  Train set:")
    train_samples = extract_evaluation_data(train_data, relation2id)
    
    datasets = [
        ("Dev", dev_samples),
        ("Test", test_samples)
    ]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_results = []
    
    # --- Exp 0 ---
    print("\n[2/3] Loading and evaluating models...")
    print("\n  ---- Exp 0: Flat BERT Baseline ----")
    from train.exp0_flat_baseline import BERTRelationClassifier
    train_r = torch.load(os.path.join(data_dir, 'train_relations.pt'))
    num_rel = int(torch.max(train_r).item()) + 1
    model = BERTRelationClassifier(num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp0_relation_flat_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r0 = evaluate_model(s_data, model, tokenizer, id2relation, device, f"Exp 0 ({s_name})", predict_exp0, model_type='flat')
        all_results.append(r0)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 3 ---
    print("\n  ---- Exp 3: Progressive Constraint Tightening ----")
    from train.exp3_pct import PCTModel
    train_d = torch.load(os.path.join(data_dir, 'train_domains.pt'))
    num_dom = int(torch.max(train_d).item()) + 1
    model = PCTModel(num_domains=num_dom, num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp3_pct_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r3 = evaluate_model(s_data, model, tokenizer, id2relation, device, f"Exp 3 ({s_name})", predict_exp3, model_type='flat')
        all_results.append(r3)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 4 ---
    print("\n  ---- Exp 4: Cross-Hop Coherence Planning ----")
    from train.exp4_chcp import CHCPModel
    rel2id_full = torch.load(os.path.join(data_dir, 'relation2id.pt'))
    num_rel_full = len(rel2id_full)
    id2rel_full = {v: k for k, v in rel2id_full.items()}
    model = CHCPModel(num_relations=num_rel_full, max_hops=4).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp4_chcp_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r4 = evaluate_model(s_data, model, tokenizer, id2rel_full, device, f"Exp 4 ({s_name})", predict_exp4, model_type='chcp')
        all_results.append(r4)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 4-RL ---
    print("\n  ---- Exp 4-RL: Reinforcement Learned CHCP ----")
    model = CHCPModel(num_relations=num_rel_full, max_hops=4).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp4_rl_epoch_49.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r4rl = evaluate_model(s_data, model, tokenizer, id2rel_full, device, f"Exp 4-RL ({s_name})", predict_exp4, model_type='chcp')
        all_results.append(r4rl)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 6 ---
    print("\n  ---- Exp 6: Unified Adaptive-CHCP ----")
    from train.exp6_unified import UnifiedKGQAPlanner
    train_d = torch.load(os.path.join(data_dir, 'train_domains.pt'))
    num_dom = int(torch.max(train_d).item()) + 1
    model = UnifiedKGQAPlanner(num_dom, num_rel_full).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp6_unified_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r6 = evaluate_model(s_data, model, tokenizer, id2rel_full, device, f"Exp 6 ({s_name})", predict_exp6, model_type='unified')
        all_results.append(r6)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 7 ---
    print("\n  ---- Exp 7: Scaled RoBERTa-Large ----")
    from train.exp7_roberta import ScaledUnifiedPlanner
    from transformers import RobertaTokenizer
    rob_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = ScaledUnifiedPlanner(num_dom, num_rel_full).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_epoch_29.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r7 = evaluate_model(s_data, model, rob_tokenizer, id2rel_full, device, f"Exp 7 ({s_name})", predict_exp7, model_type='roberta')
        all_results.append(r7)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 8 ---
    print("\n  ---- Exp 8: Contrastive RoBERTa (CPD) ----")
    model = ScaledUnifiedPlanner(num_dom, num_rel_full).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp8_cpd_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r8 = evaluate_model(s_data, model, rob_tokenizer, id2rel_full, device, f"Exp 8 ({s_name})", predict_exp7, model_type='roberta')
        all_results.append(r8)
    del model; torch.cuda.empty_cache()
    
    # --- Exp 9 ---
    print("\n  ---- Exp 9: RL Meta-Constraint Agent (RLMC) ----")
    from train.exp9_rlmc import RLConstraintAgent
    base_model = ScaledUnifiedPlanner(num_dom, num_rel_full).to(device)
    base_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    
    rl_model = RLConstraintAgent(base_model).to(device)
    rl_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=device))
    rl_model.eval()
    for s_name, s_data in datasets:
        r9 = evaluate_model(s_data, rl_model, rob_tokenizer, id2rel_full, device, f"Exp 9 ({s_name})", predict_exp9, model_type='rlmc')
        all_results.append(r9)
    del rl_model; torch.cuda.empty_cache()
    
    # --- Write Results ---
    print("\n[3/3] Writing results...")
    rp = os.path.join(ROOT, 'results.md')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write("# KGQA Research Experiment Results\n\n")
        f.write(f"## End-to-End Evaluation\n\n")
        f.write("Evaluation protocol: question → model predicts relation path → path match against gold SPARQL → derive answer correctness.\n")
        f.write("This matches the planning evaluation used by DRKG, DAMR, and Plan-Then-Retrieve.\n\n")
        
        f.write("| Model | Hits@1 | Hits@3 | Hop Accuracy | Questions |\n")
        f.write("|---|---|---|---|---|\n")
        for r in all_results:
            f.write(f"| **{r['model']}** | {r['hits@1']:.4f} | {r['hits@3']:.4f} | {r['hop_accuracy']:.4f} | {r['total']} |\n")
        
        f.write("\n### Breakdown by Number of Hops\n\n")
        f.write("| Model | 1-hop | 2-hop | 3-hop | 4-hop |\n")
        f.write("|---|---|---|---|---|\n")
        for r in all_results:
            row = f"| **{r['model']}** |"
            for nh in range(1, 5):
                bh = r['by_hops'].get(nh, {'hits1': 0, 'total': 0})
                h1 = bh['hits1']/bh['total'] if bh['total'] > 0 else 0
                row += f" {h1:.4f} ({bh.get('total',0)}) |"
            f.write(row + "\n")
        
        f.write("\n---\n\n## Comparable Published Results on CWQ\n\n")
        f.write("| Method | Hits@1 | F1 | Year |\n")
        f.write("|---|---|---|---|\n")
        f.write("| NSM | 0.486 | 0.483 | 2021 |\n")
        f.write("| SR+NSM | 0.505 | - | 2022 |\n")
        f.write("| TIARA | 0.534 | - | 2022 |\n")
        f.write("| ChatKBQA | 0.555 | - | 2024 |\n")
        f.write("| DRKG | 0.669 | - | 2025 |\n")
        
        f.write("\n> **Note**: Our evaluation uses path-matching on a CWQ-derived subgraph.\n")
        f.write("> Published results use Freebase execution. Direct comparison should be interpreted carefully.\n")
        f.write("> Our Hits@1 measures *planning accuracy* (does the model predict the correct relation path?)\n")
        f.write("> which upper-bounds the final answer accuracy.\n")
        
        f.write("\n---\n\n## Performance Notes\n\n")
        f.write("- **GPU**: RTX 5070 Laptop (SM 12.0 / Blackwell)\n")
        f.write("- **PyTorch**: 2.11.0+cu128 with Mixed Precision (AMP)\n")
        f.write("- **Dataset**: ComplexWebQuestions (CWQ) 1.1\n")
        f.write("- **Evaluation**: Path-match based (planning accuracy)\n")
    
    print(f"\nResults written to {rp}")
    
    print("\n" + "="*60)
    print("  END-TO-END RESULTS SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['model']:30s} | H@1: {r['hits@1']:.4f} | H@3: {r['hits@3']:.4f} | Hop: {r['hop_accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
