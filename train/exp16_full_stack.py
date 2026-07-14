import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Models ---
class PathAwareRanker(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fuse = nn.Linear(self.encoder.config.hidden_size * 3, 1)
    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        q_emb = self.encoder(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p_emb = self.encoder(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e_emb = self.encoder(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q_emb, p_emb, e_emb], dim=-1)).squeeze(-1)

# --- Dataset for all stages ---
class CascadingDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- Stage 1: Bi-Encoder (Fast Semantic Pruning) ---
class BiEncoderTrainer:
    def __init__(self, device):
        self.device = device
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

    def train(self, train_loader, epochs=5):
        print("\n[Stage 1] Training Bi-Encoder...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"S1-Epoch {epoch+1}")
            for item_batch in pbar:
                qs = []; ents = []; labels = []
                for item in item_batch:
                    golds = [c for c in item['candidates'] if c['is_gold']]
                    negs = [c for c in item['candidates'] if not c['is_gold']]
                    if not golds: continue
                    qs.append(item['question']); ents.append(golds[0]['name']); labels.append(1)
                    if negs: qs.append(item['question']); ents.append(random.choice(negs)['name']); labels.append(0)
                if not qs: continue
                q_enc = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt').to(self.device)
                e_enc = self.tokenizer(ents, padding=True, truncation=True, return_tensors='pt').to(self.device)
                q_emb = self.model(**q_enc).last_hidden_state[:, 0, :]
                e_emb = self.model(**e_enc).last_hidden_state[:, 0, :]
                cos_sim = F.cosine_similarity(q_emb, e_emb)
                loss = F.mse_loss(cos_sim, torch.tensor(labels, dtype=torch.float32).to(self.device))
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item(); pbar.set_postfix(loss=total_loss/len(pbar))
            torch.save(self.model.state_dict(), os.path.join(ROOT, 'checkpoints/exp16_s1_bi.pt'))

# --- Stage 2: Path-Aware Ranker ---
class Stage2Trainer:
    def __init__(self, device):
        self.device = device
        self.model = PathAwareRanker().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)

    def train(self, train_loader, epochs=5):
        print("\n[Stage 2] Training Path-Sieve...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"S2-Epoch {epoch+1}")
            for item_batch in pbar:
                qs = []; ps = []; es = []; labels = []
                for item in item_batch:
                    golds = [c for c in item['candidates'] if c['is_gold']]
                    negs = [c for c in item['candidates'] if not c['is_gold']]
                    if not golds: continue
                    # Positive
                    qs.append(item['question']); ps.append(item['path']); es.append(golds[0]['name']); labels.append(1.0)
                    # Negative
                    if negs:
                        neg = random.choice(negs)
                        qs.append(item['question']); ps.append(item['path']); es.append(neg['name']); labels.append(0.0)
                
                if not qs: continue
                q_enc = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt').to(self.device)
                p_enc = self.tokenizer(ps, padding=True, truncation=True, return_tensors='pt').to(self.device)
                e_enc = self.tokenizer(es, padding=True, truncation=True, return_tensors='pt').to(self.device)
                
                scores = self.model(q_enc['input_ids'], q_enc['attention_mask'],
                                    p_enc['input_ids'], p_enc['attention_mask'],
                                    e_enc['input_ids'], e_enc['attention_mask'])
                loss = F.binary_cross_entropy_with_logits(scores, torch.tensor(labels).to(self.device))
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item(); pbar.set_postfix(loss=total_loss/len(pbar))
            torch.save(self.model.state_dict(), os.path.join(ROOT, 'checkpoints/exp16_cds_epoch_5.pt'))

# --- Stage 3: Cross-Encoder ---
class CrossEncoderTrainer:
    def __init__(self, device):
        self.device = device
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1).to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def train(self, train_loader, epochs=5):
        print("\n[Stage 3] Training Cross-Encoder...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"S3-Epoch {epoch+1}")
            for item_batch in pbar:
                texts = []; labels = []
                for item in item_batch:
                    golds = [c for c in item['candidates'] if c['is_gold']]
                    negs = [c for c in item['candidates'] if not c['is_gold']]
                    if not golds: continue
                    texts.append((item['question'], golds[0]['name'])); labels.append(1.0)
                    if negs: texts.append((item['question'], random.choice(negs)['name'])); labels.append(0.0)
                if not texts: continue
                enc = self.tokenizer([t[0] for t in texts], [t[1] for t in texts], padding=True, truncation=True, return_tensors='pt').to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, torch.tensor(labels).to(self.device))
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item(); pbar.set_postfix(loss=total_loss/len(pbar))
            torch.save(self.model.state_dict(), os.path.join(ROOT, f'checkpoints/exp16_s3_cross.pt'))

def train_full_stack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = os.path.join(ROOT, 'data/exp16_cds_train_full.json')
    dataset = CascadingDataset(train_file)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
    
    # Train all 3 stages sequentially on full dataset
    BiEncoderTrainer(device).train(train_loader)
    Stage2Trainer(device).train(train_loader)
    CrossEncoderTrainer(device).train(train_loader)
    print("\n[Exp16] ALL 3 STAGES TRAINED ON FULL DATASET!")

if __name__ == "__main__":
    train_full_stack()
