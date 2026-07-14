"""
Exp 15: Stage 3 — Cross-Encoder Answer Selector
================================================

Fine-tunes a fast cross-encoder (MiniLM) to score (question, entity_name) pairs.
This replaces random selection from the final answer set with a semantically-aware choice.
"""

import os, sys, json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

class AnswerSelectorDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q = item['question']
        ent = item['entity_name']
        label = float(item['label'])
        
        # Cross-encoder formatting: [CLS] question [SEP] entity [SEP]
        enc = self.tokenizer(
            q,
            ent,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor([label], dtype=torch.float)
        }

def train_answer_selector():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[Selector] Using device:", device)
    
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"[Selector] Initializing {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    
    data_path = os.path.join(ROOT, 'data/exp15_selector_train_data.json')
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}. Run scripts/gen_selector_data.py first.")
        return

    train_ds = AnswerSelectorDataset(data_path, tokenizer)
    # Batch size 32 is fine for MiniLM on 8GB VRAM
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"[Selector] Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in t_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            t_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")
        
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    save_path = os.path.join(ROOT, 'checkpoints/exp15_answer_selector.pt')
    torch.save(model.state_dict(), save_path)
    print(f"[Selector] Saved model to {save_path}")

if __name__ == '__main__':
    train_answer_selector()
