"""
Exp 36: Train Stage 3 Listwise Contrastive Cross-Encoder (InfoNCE)
==================================================================

Trains `BAAI/bge-reranker-base` using Listwise Contrastive / InfoNCE Loss.
Reads the candidates directly from the existing Exp34 dataset.
For each question, groups 1 Gold + 15 Negatives.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from cds_pipeline.utils import path_to_nl

TRAIN_FILE = os.path.join(ROOT, "data/exp34_s3_listwise_train.json")
CKPT_OUT   = os.path.join(ROOT, "checkpoints/exp36_s3_infonce.pt")
MODEL_NAME = "BAAI/bge-reranker-base"
MAX_LEN    = 256
MAX_NEGS   = 15

class InfoNCEDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        import random
        random.seed(42)
        
        self.groups = []
        for item in tqdm(data, desc="Loading dataset"):
            q = item.get("question", "")
            global_path = item.get("path", "")
            
            cands = item.get("candidates", [])
            golds = [c for c in cands if c.get("is_gold")]
            negs = [c for c in cands if not c.get("is_gold")]
            
            if not golds:
                continue
                
            # Keep 1 gold
            gold = golds[0]
            
            # Subsample negatives up to MAX_NEGS
            selected_negs = random.sample(negs, min(MAX_NEGS, len(negs)))
            
            # The group always has the gold answer at index 0
            group_cands = [gold] + selected_negs
            
            cand_texts = []
            for c in group_cands:
                name = c.get("name", "").strip() or "[UNK]"
                c_path = c.get("path") or global_path or ""
                path_nl = path_to_nl(c_path)
                
                # Cand Text: Name + Path
                cand_text = f"{name} | {path_nl}" if path_nl else name
                cand_texts.append(cand_text)
                
            self.groups.append({
                "question": q,
                "cand_texts": cand_texts
            })

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]

def collate_fn(batch, tokenizer):
    questions_flat = []
    cand_texts_flat = []
    
    # Track the number of candidates per question in case some have fewer than 1+MAX_NEGS
    num_cands_per_q = []
    
    for x in batch:
        c_texts = x["cand_texts"]
        questions_flat.extend([x["question"]] * len(c_texts))
        cand_texts_flat.extend(c_texts)
        num_cands_per_q.append(len(c_texts))
        
    enc = tokenizer(
        questions_flat,
        cand_texts_flat,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return enc, num_cands_per_q

def train(model, tokenizer, dataset, device, epochs=3, lr=1e-5, batch_size=2, accum_steps=4, start_epoch=0):
    from functools import partial
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=True
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"[Exp36] Training on {len(dataset)} question groups for {epochs} epochs")
    
    for ep in range(start_epoch, epochs):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        
        optimizer.zero_grad()
        
        for step, (enc, num_cands_per_q) in enumerate(pbar):
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with autocast("cuda", dtype=torch.float16):
                outputs = model(**enc)
                logits_flat = outputs.logits.squeeze(-1)  # [sum(num_cands)]
                
                loss = 0.0
                start_idx = 0
                for num_c in num_cands_per_q:
                    # Extract the logits for this question's candidates
                    q_logits = logits_flat[start_idx : start_idx + num_c].unsqueeze(0) # [1, num_c]
                    # The gold candidate is always at index 0 for each question
                    q_target = torch.tensor([0], dtype=torch.long, device=device)
                    
                    loss += criterion(q_logits, q_target)
                    start_idx += num_c
                
                # Average loss over the batch of questions
                loss = loss / len(num_cands_per_q)
                loss = loss / accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * accum_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
            
        print(f"  Epoch {ep+1} avg loss: {total_loss / max(n_batches, 1):.4f}")
        
        ep_ckpt = CKPT_OUT.replace(".pt", f"_ep{ep+1}.pt")
        os.makedirs(os.path.dirname(ep_ckpt), exist_ok=True)
        torch.save(model.state_dict(), ep_ckpt)
        
    torch.save(model.state_dict(), CKPT_OUT)
    print(f"\n[Exp36] Final checkpoint saved: {CKPT_OUT}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2) # 2 questions = 32 pairs per forward pass
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp36] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

    if args.resume_from:
        print(f"[Exp36] Resuming from checkpoint: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))

    dataset = InfoNCEDataset(TRAIN_FILE)
    train(model, tokenizer, dataset, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, start_epoch=args.start_epoch)

if __name__ == "__main__":
    main()
