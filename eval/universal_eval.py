"""
eval/universal_eval.py

Evaluates the Exp 10 Universal Planner across three datasets:
1. ComplexWebQuestions (CWQ) - Freebase
2. WebQSP - Freebase
3. MetaQA - Movie KG

Usage: python eval/universal_eval.py --ckpt checkpoints/exp10_universal_final.pt
"""
import os, sys, torch, json, functools
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp10_universal import UniversalPlanner, UniversalDataset, collate_universal, DATASET_IDS
# We reuse CWQ/WebQSP execution logic
# For MetaQA, we use simple BFS traversal

def evaluate_on_dataset(model, loader, device, dataset_name, id2rel):
    model.eval()
    total = 0
    correct = 0
    
    t_bar = tqdm(loader, desc=f"Evaluating {dataset_name}")
    
    # Optional execution-based evaluation
    # For now, we do relation-path accuracy (Hits@1 on path predicted)
    # Strict execution evaluation follows if path matches
    
    for enc, doms, paths, nums, ds_ids in t_bar:
        enc = enc.to(device); ds_ids = ds_ids.to(device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model(enc['input_ids'], enc['attention_mask'], ds_ids)
            # [B, H, num_rel]
            pred_rels = out['rel_logits'].argmax(dim=-1)
            
        B = paths.size(0)
        for b in range(B):
            target_path = paths[b, :nums[b]].tolist()
            pred_path = pred_rels[b, :nums[b]].tolist()
            
            if target_path == pred_path:
                correct += 1
            total += 1
            
        t_bar.set_postfix(acc=f"{100*correct/total:.2f}%")
        
    acc = correct / total if total > 0 else 0
    print(f"\n[Result] {dataset_name} Path Accuracy (Strict Hits@1): {100*acc:.2f}%")
    return acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join(ROOT, 'data/processed_universal')
    
    relation2id = torch.load(os.path.join(proc_dir, 'relation2id.pt'))
    domain2id = torch.load(os.path.join(proc_dir, 'domain2id.pt'))
    id2rel = {v: k for k, v in relation2id.items()}
    
    num_rel = max(relation2id.values()) + 1
    num_dom = max(domain2id.values()) + 1
    
    model = UniversalPlanner(num_dom, num_rel).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    ckpt_num_dom = state_dict['domain_head.weight'].shape[0] if 'domain_head.weight' in state_dict else num_dom
    if ckpt_num_dom != num_dom:
        model.domain_head.weight.data[:ckpt_num_dom] = state_dict['domain_head.weight']
        model.domain_head.bias.data[:ckpt_num_dom] = state_dict['domain_head.bias']
        del state_dict['domain_head.weight']
        del state_dict['domain_head.bias']
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    collate = functools.partial(collate_universal, tokenizer=tokenizer)
    
    results = {}
    
    # 1. Evaluate CWQ Dev
    cwq_ds = UniversalDataset(os.path.join(proc_dir, 'cwq/dev.json'), relation2id, domain2id, 'cwq')
    cwq_loader = DataLoader(cwq_ds, batch_size=64, collate_fn=collate)
    results['cwq'] = evaluate_on_dataset(model, cwq_loader, device, 'CWQ-Dev', id2rel)
    
    # 2. Evaluate WebQSP Test
    webq_ds = UniversalDataset(os.path.join(proc_dir, 'webqsp/test.json'), relation2id, domain2id, 'webqsp')
    webq_loader = DataLoader(webq_ds, batch_size=64, collate_fn=collate)
    results['webqsp'] = evaluate_on_dataset(model, webq_loader, device, 'WebQSP-Test', id2rel)
    
    # 3. Evaluate MetaQA Test (all hops)
    meta_ds = UniversalDataset(os.path.join(proc_dir, 'metaqa/test.json'), relation2id, domain2id, 'metaqa')
    meta_loader = DataLoader(meta_ds, batch_size=128, collate_fn=collate)
    results['metaqa'] = evaluate_on_dataset(model, meta_loader, device, 'MetaQA-Test', id2rel)
    
    print("\n" + "="*40)
    print("FINAL UNIVERSAL SCORES (Path Hits@1)")
    print("="*40)
    for ds, score in results.items():
        print(f"{ds.upper():<10}: {100*score:>6.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()
