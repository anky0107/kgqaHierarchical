import os, sys, torch, functools
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp6_unified import UnifiedDataset, collate_unified
from train.exp7_roberta import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent

def calculate_metrics(target_path, logits_seq, actions_seq, domain_pred, domain_target):
    """
    Calculates Precision, Recall, and Hits@1 based on the RL bounding action constraint.
    For each hop:
    Action 0 (TIGHT): K=1
    Action 1 (MEDIUM): K=5
    Action 2 (LOOSE): K=50 (Approximating domain match)
    Action 3 (STOP): K=0
    """
    target_path = target_path.tolist()
    L = len(target_path)
    
    total_retrieved = 0
    total_relevant = L
    total_retrieved_relevant = 0
    
    hits_at_1 = True
    
    for h in range(L):
        # We need to evaluate the model's choices against the true length L
        action = actions_seq[h] if h < len(actions_seq) else 3
        logits = logits_seq[h]
        gold = target_path[h]
        
        # Hits@1 checks strictly rank 1
        if torch.argmax(logits).item() != gold:
            hits_at_1 = False
            
        if action == 0: # TIGHT
            retrieved = torch.topk(logits, 1).indices.tolist()
            total_retrieved += 1
            if gold in retrieved:
                total_retrieved_relevant += 1
                
        elif action == 1: # MEDIUM
            retrieved = torch.topk(logits, 5).indices.tolist()
            total_retrieved += 5
            if gold in retrieved:
                total_retrieved_relevant += 1
                
        elif action == 2: # LOOSE (Domain logic)
            total_retrieved += 50
            if domain_pred == domain_target:
                total_retrieved_relevant += 1
                
        elif action == 3: # STOP
            pass # K=0, so no retrieval
            
    # Precision = retrieved_relevant / total_retrieved
    precision = total_retrieved_relevant / total_retrieved if total_retrieved > 0 else 0
    # Recall = retrieved_relevant / total_relevant
    recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
    
    return hits_at_1, precision, recall

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/exp9_rlmc_epoch_9.pt')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Exp 9 RL Meta-Constraint on {device}")
    
    rel2id = torch.load('data/processed_entity/relation2id.pt')
    dom2id = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id); num_dom = len(dom2id)
    
    # Load Base Unified Planner
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base_model.load_state_dict(torch.load('checkpoints/exp7_roberta_best.pt', map_location=device))
    base_model.eval()
    
    # Load RL Header Agent
    model = RLConstraintAgent(base_model).to(device)
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print("Loaded RL Weights successfully.")
    except:
        print("Could not load RL weights. Testing architecture natively.")
    model.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    collate = functools.partial(collate_unified, tokenizer=tokenizer)
    
    test_ds = UnifiedDataset('data/cwq_test.json', rel2id, dom2id)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate)
    
    action_counts = {0:0, 1:0, 2:0, 3:0}
    
    total_hits1 = 0
    total_prec = 0
    total_rec = 0
    samples = 0
    
    t_bar = tqdm(test_loader, desc="CWQ RL Evaluation")
    
    for enc, doms, paths, nums in t_bar:
        enc = enc.to(device); doms = doms.to(device)
        B = paths.size(0)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            action_logits, state_values, rel_logits, domain_logits = model(enc['input_ids'], enc['attention_mask'])
            
            # Get the predicted actions
            actions = torch.argmax(action_logits, dim=-1) # [B, max_hops]
            
        for b in range(B):
            L = int(nums[b].item())
            target_path = paths[b, :L]
            
            dom_pred = torch.argmax(domain_logits[b]).item()
            dom_true = doms[b].item()
            
            act_seq = actions[b].tolist()
            log_seq = rel_logits[b]
            
            hit1, p, r = calculate_metrics(target_path, log_seq, act_seq, dom_pred, dom_true)
            
            # Track Action behaviors
            for h in range(L):
                a = act_seq[h] if h < len(act_seq) else 3
                action_counts[a] += 1
                
            if hit1: total_hits1 += 1
            total_prec += p
            total_rec += r
            samples += 1
            
    avg_hits = total_hits1 / samples
    avg_prec = total_prec / samples
    avg_rec = total_rec / samples
    avg_f1 = (2 * avg_prec * avg_rec) / (avg_prec + avg_rec) if (avg_prec + avg_rec) > 0 else 0
    
    print("\n" + "="*50)
    print("EXPERIMENT 9: METRICS AND STATISTICS (CWQ Test)")
    print("="*50)
    print(f"Hits@1 (Strict Path Accuracy) : {100*avg_hits:.2f}%")
    print(f"Precision (Efficiency)        : {100*avg_prec:.2f}%")
    print(f"Recall (Coverage Tolerance)   : {100*avg_rec:.2f}%")
    print(f"Macro F1-Score                : {100*avg_f1:.2f}%")
    print("-" * 50)
    print("Action Space Distribution:")
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        print(f"  TIGHT (Top-1)   : {100 * action_counts[0] / total_actions:.2f}%")
        print(f"  MEDIUM (Top-5)  : {100 * action_counts[1] / total_actions:.2f}%")
        print(f"  LOOSE (Domain)  : {100 * action_counts[2] / total_actions:.2f}%")
        print(f"  STOP (Prune)    : {100 * action_counts[3] / total_actions:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
