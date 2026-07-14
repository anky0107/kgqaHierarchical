import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CDSDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128, neg_samples=15):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q = item['question']
        path_str = " -> ".join(["/".join(p[:2]) for p in item['path']]) # simplified path
        
        golds = [c for c in item['candidates'] if c['is_gold']]
        negs = [c for c in item['candidates'] if not c['is_gold']]
        
        if not golds: # fallback if no gold in beam
            return None
            
        pos = golds[0]
        # Sample hard negatives from the beam
        import random
        if len(negs) > self.neg_samples:
            sampled_negs = random.sample(negs, self.neg_samples)
        else:
            sampled_negs = negs
            
        return {
            'q': q,
            'path': path_str,
            'pos': pos['name'],
            'negs': [n['name'] for n in sampled_negs]
        }

class PathAwareRanker(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Simple attention layer to fuse Path and Entity
        self.fuse = nn.Linear(self.hidden_dim * 3, 1)

    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        q_emb = self.encoder(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :] # [B, D]
        p_emb = self.encoder(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :] # [B, D]
        e_emb = self.encoder(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :] # [B, D]
        
        combined = torch.cat([q_emb, p_emb, e_emb], dim=-1)
        score = self.fuse(combined).squeeze(-1)
        return score

def collate_cds(batch, tokenizer, max_len):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    questions = [b['q'] for b in batch]
    paths = [b['path'] for b in batch]
    
    # Each batch has 1 pos and N negs
    # We will flatten them for the encoder
    all_entities = []
    for b in batch:
        all_entities.append(b['pos'])
        all_entities.extend(b['negs'])
    
    # Tile questions and paths to match entities
    q_expanded = []
    p_expanded = []
    for b in batch:
        count = 1 + len(b['negs'])
        q_expanded.extend([b['q']] * count)
        p_expanded.extend([b['path']] * count)
        
    q_enc = tokenizer(q_expanded, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    p_enc = tokenizer(p_expanded, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    e_enc = tokenizer(all_entities, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    
    # labels: for each sample in batch, index 0 is positive
    return q_enc, p_enc, e_enc, len(batch), [1 + len(b['negs']) for b in batch]

def train_exp16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp16] Training on {device}")
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PathAwareRanker(model_name).to(device)
    
    train_file = os.path.join(ROOT, 'data/exp16_cds_train.json')
    dev_file = os.path.join(ROOT, 'data/exp16_cds_dev.json')
    
    # Wait for data collector if necessary
    while not os.path.exists(train_file):
        print("Waiting for data collector...")
        time.sleep(30)
        
    train_ds = CDSDataset(train_file, tokenizer)
    dev_ds = CDSDataset(dev_file, tokenizer)
    
    import functools
    collate = functools.partial(collate_cds, tokenizer=tokenizer, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            if batch is None: continue
            q_enc, p_enc, e_enc, batch_size, group_sizes = batch
            
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            p_enc = {k: v.to(device) for k, v in p_enc.items()}
            e_enc = {k: v.to(device) for k, v in e_enc.items()}
            
            scores = model(q_enc['input_ids'], q_enc['attention_mask'], 
                           p_enc['input_ids'], p_enc['attention_mask'],
                           e_enc['input_ids'], e_enc['attention_mask'])
            
            # Contrastive loss (InfoNCE style) per group
            loss = 0
            start = 0
            for sz in group_sizes:
                group_scores = scores[start:start+sz]
                # Index 0 is always positive
                loss += F.cross_entropy(group_scores.unsqueeze(0), torch.tensor([0]).to(device))
                start += sz
            
            loss = loss / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/len(pbar))
            
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(ROOT, f'checkpoints/exp16_cds_epoch_{epoch+1}.pt'))

if __name__ == "__main__":
    import time
    train_exp16()
