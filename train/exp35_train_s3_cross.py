"""
Exp 35: Train Stage 3 Pointwise Cross-Encoder (BGE)
===================================================

Trains `BAAI/bge-reranker-base` using BCEWithLogitsLoss.
Reads the candidates directly from the existing Exp34 dataset.
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
CKPT_OUT   = os.path.join(ROOT, "checkpoints/exp35_s3_cross.pt")
MODEL_NAME = "BAAI/bge-reranker-base"
MAX_LEN    = 256

class CrossEncoderDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        import random
        random.seed(42)
        
        self.pairs = []
        for item in tqdm(data, desc="Loading dataset"):
            q = item.get("question", "")
            global_path = item.get("path", "")
            
            cands = item.get("candidates", [])
            golds = [c for c in cands if c.get("is_gold")]
            negs = [c for c in cands if not c.get("is_gold")]
            
            if not golds:
                continue
                
            # Keep 1 gold and up to 5 random negatives
            selected_cands = golds + random.sample(negs, min(5, len(negs)))
            
            for c in selected_cands:
                name = c.get("name", "").strip() or "[UNK]"
                c_path = c.get("path") or global_path or ""
                path_nl = path_to_nl(c_path)
                
                # Cand Text: Name + Path
                cand_text = f"{name} | {path_nl}" if path_nl else name
                
                label = 1.0 if c.get("is_gold") else 0.0
                
                self.pairs.append({
                    "question": q,
                    "cand_text": cand_text,
                    "label": label
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch, tokenizer):
    questions = [x["question"] for x in batch]
    cand_texts = [x["cand_text"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    
    enc = tokenizer(
        questions,
        cand_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return enc, labels

def train(model, tokenizer, dataset, device, epochs=3, lr=1e-5, batch_size=32, accum_steps=2):
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
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    print(f"[Exp35] Training on {len(dataset)} pairs for {epochs} epochs (Batch: {batch_size * accum_steps})")
    
    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        
        optimizer.zero_grad()
        
        for step, (enc, labels) in enumerate(pbar):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            
            with autocast("cuda", dtype=torch.float16):
                outputs = model(**enc)
                logits = outputs.logits.squeeze(-1)  # [B]
                loss = criterion(logits, labels) / accum_steps
                
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
    print(f"\n[Exp35] Final checkpoint saved: {CKPT_OUT}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp35] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

    dataset = CrossEncoderDataset(TRAIN_FILE)
    train(model, tokenizer, dataset, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
