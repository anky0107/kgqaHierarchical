import os, sys, json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

class PathRerankerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for item in self.data:
            q = item['question']
            
            # Positive
            pos_path = " -> ".join(item['positive_path'])
            self.samples.append((q, pos_path, 1.0))
            
            # Negatives
            for neg in item['negative_paths']:
                neg_path = " -> ".join(neg)
                self.samples.append((q, neg_path, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, path_str, label = self.samples[idx]
        
        # Cross-encoder formatting: [CLS] question [SEP] path [SEP]
        enc = self.tokenizer(
            q,
            path_str,
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

def train_reranker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing Exp9 Path Re-ranker Training...")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1).to(device)
    
    train_ds = PathRerankerDataset(os.path.join(ROOT, 'data/exp9_reranker_train_data.json'), tokenizer)
    # Reducing batch size to 4 to fit in 8GB VRAM as per prev experiments
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    accumulation_steps = 8 # Effective batch size 32
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=(len(train_loader)//accumulation_steps)*3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        optimizer.zero_grad()
        for i, batch in enumerate(t_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
            t_bar.set_postfix({'loss': f"{loss.item()*accumulation_steps:.4f}"})
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")
        
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, 'checkpoints/exp9_reranker_final.pt'))
    print("Saved exp9_reranker_final.pt")

if __name__ == '__main__':
    train_reranker()
