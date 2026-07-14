"""
Exp 16 v3: CDS Trainer with PATH-AWARE Stage 3 (Listwise Ranking)
================================================================
Key upgrades from v2:
  - Stage 3 (Cross-Encoder) is now PATH-AWARE.
  - Input sequence: [Question + Path] [Entity]
  - This allows the final judge to use the model's reasoning logic to disambiguate.
  - Uses KL-Distillation loss (v2 ablation winner).
"""
import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F, random
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Loss Functions ──

def loss_kl_distill(scores):
    """
    KL-Divergence distillation — v2 winner.
    Gold gets soft label=1.0; negatives share 0.1 uniformly.
    """
    N = scores.shape[0]
    teacher = torch.full((N,), 0.1 / max(N - 1, 1), device=scores.device)
    teacher[0] = 1.0
    teacher = teacher / teacher.sum()
    return F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction='sum')

def loss_soft_margin(logits):
    """Used for Stage 2 (Path-Sieve) as it handles multi-gold answers gracefully."""
    labels = torch.zeros_like(logits); labels[0] = 1.0
    return nn.MultiLabelSoftMarginLoss()(logits.unsqueeze(0), labels.unsqueeze(0))

class PathAwareRanker(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        q = self.encoder(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p = self.encoder(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e = self.encoder(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q, p, e], dim=-1)).squeeze(-1)

# ── Dataset ──────────────────────────────────────────────────────────────────
class ListwiseDataset(Dataset):
    def __init__(self, path, max_samples=None):
        print(f"[Dataset] Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        if max_samples:
            raw = raw[:max_samples]
            
        self.samples = []
        for s in raw:
            if not any(c['is_gold'] for c in s['candidates']):
                continue
            
            # Robustness: Ensure path is a string
            path_val = s.get('path', '')
            if isinstance(path_val, list):
                # Flatten list of lists if needed
                if path_val and isinstance(path_val[0], list):
                    path_val = " -> ".join([r for sub in path_val for r in sub])
                else:
                    path_val = " -> ".join(path_val)
            s['path'] = str(path_val)
            self.samples.append(s)
            
        print(f"[Dataset] Loaded {len(self.samples)} valid samples.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def listwise_collate(batch): return batch

# ── Stage 1: Bi-Encoder (Fast Pruning) ──────────────────────────────────────
class Stage1Trainer:
    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.opt = AdamW(self.model.parameters(), lr=2e-5)

    def train(self, loader, epochs=3, accum_steps=4):
        print("\n[S1] Training Bi-Encoder (MSE cosine)...")
        scaler = GradScaler('cuda')
        for ep in range(epochs):
            self.model.train(); total = 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"S1 Ep{ep+1}")
            for step, batch in enumerate(pbar):
                qs, ents, labels = [], [], []
                for item in batch:
                    golds = [c for c in item['candidates'] if c['is_gold']]
                    negs  = [c for c in item['candidates'] if not c['is_gold']]
                    if not golds: continue
                    qs.append(item['question']); ents.append(golds[0]['name']); labels.append(1.0)
                    for neg in random.sample(negs, min(3, len(negs))):
                        qs.append(item['question']); ents.append(neg['name']); labels.append(0.0)
                if not qs: continue
                qe = self.tok(qs, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
                ee = self.tok(ents, padding=True, truncation=True, max_length=64, return_tensors='pt').to(self.device)
                with autocast('cuda'):
                    qv = self.model(**qe).last_hidden_state[:, 0, :]
                    ev = self.model(**ee).last_hidden_state[:, 0, :]
                    loss = F.mse_loss(F.cosine_similarity(qv, ev), torch.tensor(labels).to(self.device)) / accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(self.opt); scaler.update(); self.opt.zero_grad()
                total += loss.item() * accum_steps; pbar.set_postfix(loss=f"{total/(step+1):.4f}")
        torch.save(self.model.state_dict(), os.path.join(ROOT, 'checkpoints/exp16v3_s1_bi.pt'))

# ── Stage 2: Path-Aware Sieve (Fusion) ──────────────────────────────────────
class Stage2Trainer:
    def __init__(self, device):
        self.device = device
        self.model = PathAwareRanker(model_name="sentence-transformers/all-mpnet-base-v2").to(device)
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.opt = AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-2)

    def train(self, loader, epochs=3, accum_steps=4):
        print("\n[S2] Training Path-Sieve (SoftMargin)...")
        scaler = GradScaler('cuda')
        for ep in range(epochs):
            self.model.train(); total = 0; count = 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"S2 Ep{ep+1}")
            for step, batch in enumerate(pbar):
                all_q, all_p, all_e, offsets, offset = [], [], [], [], 0
                for item in batch:
                    golds = [c for c in item['candidates'] if c['is_gold']]
                    negs  = [c for c in item['candidates'] if not c['is_gold']]
                    if not golds or not negs: continue
                    cands = golds[:1] + random.sample(negs, min(15, len(negs)))
                    N = len(cands); path = item.get('path') or ''; q = item['question']
                    all_q.extend([str(q)] * N)
                    all_p.extend([str(path)] * N)
                    all_e.extend([str(c.get('name', '')) for c in cands])
                    offsets.append((offset, offset + N)); offset += N
                if not all_q: continue
                qe = self.tok(all_q, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
                pe = self.tok(all_p, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
                ee = self.tok(all_e, padding=True, truncation=True, max_length=64, return_tensors='pt').to(self.device)
                with autocast('cuda'):
                    all_scores = self.model(qe['input_ids'], qe['attention_mask'],
                                           pe['input_ids'], pe['attention_mask'],
                                           ee['input_ids'], ee['attention_mask'])
                    loss = torch.stack([loss_soft_margin(all_scores[s:e]) for s, e in offsets]).mean() / accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(self.opt); scaler.update(); self.opt.zero_grad()
                total += loss.item() * accum_steps; count += 1; pbar.set_postfix(loss=f"{total/count:.4f}")
        torch.save(self.model.state_dict(), os.path.join(ROOT, 'checkpoints/exp16v3_s2_path.pt'))

# ── Stage 3: PATH-AWARE Cross-Encoder (Upgrade) ──────────────────────────────
class Stage3TrainerV3:
    def __init__(self, device):
        self.device = device
        # BGE-reranker-base (109M)
        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
        self.opt = AdamW(self.model.parameters(), lr=5e-6)

    def train(self, loader, epochs=5, accum_steps=16):
        print("\n[S3] Training PATH-AWARE Cross-Encoder (KL-Distill, AMP)...")
        scaler = GradScaler('cuda')
        for ep in range(epochs):
            self.model.train(); total = 0; count = 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"S3 Ep{ep+1}")
            for step, batch in enumerate(pbar):
                all_inputs, all_es, offsets, offset = [], [], [], 0
                for item in batch:
                    golds = [c for c in item['candidates'] if c['is_gold']]
                    negs  = [c for c in item['candidates'] if not c['is_gold']]
                    if not golds or not negs: continue
                    cands = golds[:1] + random.sample(negs, min(15, len(negs)))
                    q = item['question']; p = item.get('path', ''); N = len(cands)
                    # V3 UPGRADE: Combine Question and Path to inform the judge
                    # We use a simple separator that BGE can understand
                    q_with_path = f"{q} [PATH] {p}"
                    all_inputs.extend([q_with_path] * N)
                    all_es.extend([str(c.get('name', '')) for c in cands])
                    offsets.append((offset, offset + N)); offset += N
                if not all_inputs: continue
                enc = self.tok(all_inputs, all_es, padding=True, truncation=True, max_length=192, return_tensors='pt').to(self.device)
                with autocast('cuda'):
                    all_logits = self.model(**enc).logits.squeeze(-1)
                    loss = torch.stack([loss_kl_distill(all_logits[s:e]) for s, e in offsets]).mean() / accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(self.opt); scaler.update(); self.opt.zero_grad()
                total += loss.item() * accum_steps; count += 1; pbar.set_postfix(loss=f"{total/count:.4f}")
        torch.save(self.model.state_dict(), os.path.join(ROOT, 'checkpoints/exp16v3_s3_cross.pt'))

# ── Main ──────────────────────────────────────────────────────────────────────
def train_v3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the 320MB dataset for stability; ListwiseDataset handles the path conversion
    train_file = os.path.join(ROOT, 'data/exp16_cds_train.json')
    if not os.path.exists(train_file):
        train_file = os.path.join(ROOT, 'data/exp16_cds_train_full.json')
    
    if not os.path.exists(train_file):
        print(f"[ERROR] No training data found at {train_file}")
        return

    print(f"[v3] Loading dataset from {train_file}...")
    dataset = ListwiseDataset(train_file)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=listwise_collate, pin_memory=True)
    
    # Stage1Trainer(device).train(loader)
    # Stage2Trainer(device).train(loader)
    Stage3TrainerV3(device).train(loader)
    print("\n[Exp16 v3] PATH-AWARE CDS TRAINED!")

if __name__ == "__main__":
    train_v3()
