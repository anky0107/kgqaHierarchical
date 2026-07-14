"""
Exp 38: Train T5 Reranker with Custom DPO
=========================================

Applies Direct Preference Optimization (DPO) to the T5 Multiple-Choice Reranker.
Since newer versions of the `trl` library dropped support for Encoder-Decoder models,
we implement the DPO loss manually in pure PyTorch.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

MODEL_NAME = "google/flan-t5-base"
SFT_CKPT = os.path.join(ROOT, "checkpoints/exp31_t5_mc_s3.pt")
TRAIN_FILE = os.path.join(ROOT, "data/exp37_t5_dpo_train.json")
FINAL_CKPT = os.path.join(ROOT, "checkpoints/exp38_t5_dpo_s3.pt")

import argparse

class DPODataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    prompts = [x["prompt"] for x in batch]
    chosens = [x["chosen"] for x in batch]
    rejecteds = [x["rejected"] for x in batch]
    
    # Tokenize prompts
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding=True, return_tensors="pt")
    
    # Tokenize chosen targets
    chosen_labels = tokenizer(text_target=chosens, max_length=64, truncation=True, padding=True, return_tensors="pt")
    # Replace pad token id with -100 for cross entropy
    chosen_labels["input_ids"][chosen_labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    # Tokenize rejected targets
    rejected_labels = tokenizer(text_target=rejecteds, max_length=64, truncation=True, padding=True, return_tensors="pt")
    rejected_labels["input_ids"][rejected_labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "chosen_labels": chosen_labels["input_ids"],
        "rejected_labels": rejected_labels["input_ids"]
    }

def get_log_probs(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(labels.size(0), labels.size(1))
    
    pad_mask = labels != -100
    log_probs = -(loss * pad_mask).sum(dim=1)
    return log_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1, help="The beta factor in DPO loss.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--accum_steps", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading models from {SFT_CKPT}...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))
    model.to(device)
    model.train()
    
    ref_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    ref_model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))
    ref_model.to(device)
    ref_model.eval()

    dataset = DPODataset(TRAIN_FILE)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=True
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(device="cuda")
    
    print(f"Starting Custom DPO training on {len(dataset)} pairs...")
    
    for ep in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{args.epochs}")
        total_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    ref_log_prob_c = get_log_probs(ref_model, input_ids, attention_mask, chosen_labels)
                    ref_log_prob_r = get_log_probs(ref_model, input_ids, attention_mask, rejected_labels)
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                active_log_prob_c = get_log_probs(model, input_ids, attention_mask, chosen_labels)
                active_log_prob_r = get_log_probs(model, input_ids, attention_mask, rejected_labels)
                
                pi_logratios = active_log_prob_c - active_log_prob_r
                ref_logratios = ref_log_prob_c - ref_log_prob_r
                
                if torch.isnan(pi_logratios).any() or torch.isnan(ref_logratios).any():
                    print(f"NAN DETECTED!")
                    print(f"ref_c: {ref_log_prob_c}, ref_r: {ref_log_prob_r}")
                    print(f"act_c: {active_log_prob_c}, act_r: {active_log_prob_r}")
                    
                logits = pi_logratios - ref_logratios
                loss = -F.logsigmoid(args.beta * logits).mean()
                
                loss = loss / args.accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * args.accum_steps
            pbar.set_postfix(loss=f"{total_loss / max(1, (step+1)):.4f}", margins=logits.mean().item())

    print(f"\nSaving final state dict to {FINAL_CKPT}...")
    os.makedirs(os.path.dirname(FINAL_CKPT), exist_ok=True)
    torch.save(model.state_dict(), FINAL_CKPT)
    print("Done!")

if __name__ == "__main__":
    main()
