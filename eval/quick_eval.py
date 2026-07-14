"""
Quick evaluation runner for main experiments on Dev set.
"""
import json, os, sys, torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer, BertTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.sparql_parser import find_reasoning_path

def extract_evaluation_data(cwq_data, relation2id):
    samples = []
    skipped = 0
    for item in cwq_data:
        question = item['question']
        sparql = item['sparql']
        path = find_reasoning_path(sparql)
        if path is None:
            skipped += 1
            continue
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
        samples.append({
            'question': question,
            'gold_path': gold_relations,
            'num_hops': len(gold_relations),
        })
    return samples

def predict_exp7(model, tokenizer, question, device, max_hops=4, k=10):
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            rel_logits = out['rel_logits']
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        results.append({'top_ids': topk.cpu().tolist()})
    return results

def predict_exp9(rl_agent, tokenizer, question, device, max_hops=4, k=10):
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            action_logits, _, rel_logits, _ = rl_agent(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    results = []
    actions = torch.argmax(action_logits[0], dim=-1).tolist()
    for h in range(max_hops):
        a = actions[h]
        probs = F.softmax(rel_logits[0, h], dim=-1)
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
            results.append({'top_ids': topw.cpu().tolist()})
        else:
            break
    return results

def evaluate_model(samples, model, tokenizer, device, model_name, predict_fn, model_type='roberta'):
    total = 0
    hits1 = 0
    for sample in tqdm(samples, desc=f"Eval {model_name}"):
        question = sample['question']
        gold_path = sample['gold_path']
        hop_preds = predict_fn(model, tokenizer, question, device, max_hops=4, k=10)
        path_match_1 = True
        for h, gold_hop in enumerate(gold_path):
            gold_id = gold_hop['relation_id']
            if h < len(hop_preds):
                pred_top = hop_preds[h]['top_ids']
                if pred_top[0] != gold_id:
                    path_match_1 = False
            else:
                path_match_1 = False
        if path_match_1:
            hits1 += 1
        total += 1
    h1 = hits1 / total if total > 0 else 0
    print(f"  {model_name} | Hits@1: {h1:.4f} ({hits1}/{total})")
    return h1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_dir = os.path.join(ROOT, 'data/processed_entity')
    relation2id = torch.load(os.path.join(data_dir, 'relation2id.pt'))
    
    print("Loading Dev data...")
    dev_data = json.load(open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8'))
    dev_samples = extract_evaluation_data(dev_data, relation2id)
    print(f"Extracted {len(dev_samples)} Dev samples.")
    
    rob_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    num_dom = 42 # Constant domain size or load if needed
    num_rel_full = len(relation2id)
    
    # --- Exp 7 ---
    print("\nEvaluating Exp 7 (RoBERTa-Large)...")
    from train.exp7_roberta import ScaledUnifiedPlanner
    model7 = ScaledUnifiedPlanner(num_domains=70, num_relations=num_rel_full).to(device)
    model7.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    model7.eval()
    evaluate_model(dev_samples, model7, rob_tokenizer, device, "Exp 7 (Dev)", predict_exp7)
    del model7; torch.cuda.empty_cache()
    
    # --- Exp 9 ---
    print("\nEvaluating Exp 9 (RLMC)...")
    from train.exp9_rlmc import RLConstraintAgent
    base_model = ScaledUnifiedPlanner(num_domains=70, num_relations=num_rel_full).to(device)
    base_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    rl_model = RLConstraintAgent(base_model).to(device)
    rl_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=device))
    rl_model.eval()
    evaluate_model(dev_samples, rl_model, rob_tokenizer, device, "Exp 9 (Dev)", predict_exp9)

if __name__ == '__main__':
    main()
