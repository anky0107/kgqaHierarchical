"""
Exp 8: Contrastive RoBERTa (CPD)

Fine-tuning the scaled RoBERTa model with Path-Level Contrastive Hard-Negative Mining (InfoNCE).
"""
import os, sys, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp6_unified import UnifiedDataset, collate_unified
from train.exp7_roberta import ScaledUnifiedPlanner

def path_contrastive_loss(logits, gold_paths, lens, tau=0.1):
    """
    Computes InfoNCE loss between the gold path and 'Hard Negative' paths.
    Hard negatives are generated dynamically by swapping one relation in the gold
    path with the model's highest-scoring incorrect relation for that hop.
    """
    B = logits.size(0)
    losses = []
    for b in range(B):
        L = int(lens[b].item())
        if L == 0: continue
        
        # Pos score: sum of logits for the gold relation sequence
        pos_score = sum([logits[b, h, gold_paths[b, h]] for h in range(L)])
        
        # Neg scores: sum of logits for adversarial sequences
        neg_scores = []
        for h in range(L):
            hop_logits = logits[b, h]
            # Find the highest scoring relation that is NOT the gold relation
            top2 = torch.topk(hop_logits, 2).indices
            neg_r = top2[1] if top2[0] == gold_paths[b, h] else top2[0]
            
            # The score if we had taken the adversarial relation at this hop instead
            n_score = pos_score - hop_logits[gold_paths[b, h]] + hop_logits[neg_r]
            neg_scores.append(n_score)
            
        # InfoNCE
        # Stack scores: [pos, neg1, neg2, ...]
        all_scores = torch.stack([pos_score] + neg_scores) / tau
        # Maximize probability of the pos_score (index 0)
        loss_b = -F.log_softmax(all_scores, dim=0)[0]
        losses.append(loss_b)
        
    if not losses:
        return torch.tensor(0.0).to(logits.device)
    return torch.stack(losses).mean()

def train_exp8_cpd():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Maps
    rel2id = torch.load('data/processed_entity/relation2id.pt')
    dom2id = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id)
    num_dom = len(dom2id)
    
    # Init Model
    model = ScaledUnifiedPlanner(num_dom, num_rel, hidden_dim=512).to(device)
    
    start_epoch = 0
    exp8_ckpts = [f for f in os.listdir(os.path.join(ROOT, 'checkpoints')) if f.startswith('exp8_cpd_epoch_') and f.endswith('.pt')]
    
    if exp8_ckpts:
        latest_ckpt = max(exp8_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0]) + 1
        ckpt_path = os.path.join(ROOT, 'checkpoints', latest_ckpt)
        print(f"Resuming Exp 8 from {ckpt_path} (Starting at Epoch {start_epoch})")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        # Load Exp 7 Weights (The prerequisite starting point)
        ckpt_path = os.path.join(ROOT, 'checkpoints', 'exp7_roberta_epoch_29.pt')
        print(f"Loading Base Weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
    # We use a very low learning rate because we are just fine-tuning the decision boundaries
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    # Dataset
    train_ds = UnifiedDataset('data/cwq_train.json', rel2id, dom2id)
    dev_ds = UnifiedDataset('data/cwq_dev.json', rel2id, dom2id)
    collate = functools.partial(collate_unified, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=8, collate_fn=collate)
    
    epochs = 10
    best_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = 4 

    metrics_dir = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp8_cpd.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,dev_ce_loss,dev_cpd_loss\n")

    print(f"\nStarting Contrastive Fine-Tuning (Exp 8)...")
    
    # Contrastive Loss Weight
    lambda_cpd = 0.5 
    
    for epoch in range(start_epoch, epochs):
        model.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for i, (enc, doms, paths, nums) in enumerate(t_bar):
            enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)
            
            with torch.amp.autocast('cuda'):
                out = model(enc['input_ids'], enc['attention_mask'])
                
                # Standard Cross Entropy
                loss_dom = F.cross_entropy(out['domain_logits'], doms)
                loss_rel = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                
                # Stop Targets
                B, H = paths.size()
                stop_targets = torch.zeros(B, H).to(device)
                for b in range(B):
                    stop_targets[b, :nums[b]] = 1.0
                loss_stop = F.binary_cross_entropy_with_logits(out['stop_logits'], stop_targets)
                
                # Dynamic Hard Negative Contrastive Loss
                loss_cpd = path_contrastive_loss(out['rel_logits'], paths, nums, tau=0.1)
                
                total_loss = (loss_dom + loss_rel + loss_stop + (lambda_cpd * loss_cpd)) / accumulation_steps
            
            scaler.scale(total_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            t_bar.set_postfix(loss=total_loss.item() * accumulation_steps, cpd=loss_cpd.item())
            
        # Eval
        model.eval()
        v_ce_loss = 0
        v_cpd_loss = 0
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for enc, doms, paths, nums in dev_loader:
                enc = enc.to(device); paths = paths.to(device); nums = nums.to(device)
                out = model(enc['input_ids'], enc['attention_mask'])
                
                v_ce_loss += F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1)).item()
                v_cpd_loss += path_contrastive_loss(out['rel_logits'], paths, nums, tau=0.1).item()
        
        avg_ce = v_ce_loss / len(dev_loader)
        avg_cpd = v_cpd_loss / len(dev_loader)
        print(f"Epoch {epoch} | Dev CE: {avg_ce:.4f} | Dev CPD: {avg_cpd:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_ce:.4f},{avg_cpd:.4f}\n")
            
        torch.save(model.state_dict(), f'checkpoints/exp8_cpd_epoch_{epoch}.pt')
        if avg_cpd < best_loss: # Optimize based on contrastive separation!
            best_loss = avg_cpd
            torch.save(model.state_dict(), f'checkpoints/exp8_cpd_best.pt')

if __name__ == "__main__":
    train_exp8_cpd()
