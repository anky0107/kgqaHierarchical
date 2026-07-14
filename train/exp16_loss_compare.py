"""
Exp 16 v2: Loss Function Comparison (Stage 3 Ablation)
========================================================
Tests 5 loss functions from loss_functions_study.md on 500 samples / 2 epochs.
Stage 3 (Cross-Encoder) is evaluated because it has the most direct impact on Hit@1.

Loss functions tested:
  1. InfoNCE (Listwise) — current v2 approach
  2. Focal BCE — down-weights easy negatives, class-imbalance robust
  3. Multi-label Soft Margin — handles multi-gold answers
  4. KL Divergence — soft teacher distillation (cosine-sim soft labels)
  5. Triplet Margin — simpler contrastive baseline

Evaluation: After each loss-function training, run a quick Hit@1 check on 200 dev samples.
Report table → auto-select winner → print recommendation.
"""
import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# ── Loss Implementations ──────────────────────────────────────────────────────

def loss_infonce(logits, temperature=0.07):
    """Listwise InfoNCE: gold is always index 0."""
    return F.cross_entropy(logits.unsqueeze(0) / temperature,
                           torch.zeros(1, dtype=torch.long, device=logits.device))

def loss_focal_bce(logits, gamma=2.0, alpha=0.25):
    """Focal BCE: gold=1 at idx 0, rest=0. Downweights easy negatives."""
    labels = torch.zeros_like(logits); labels[0] = 1.0
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    p_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()

def loss_multilabel_soft_margin(logits):
    """Multi-label Soft Margin: gold=1 at idx 0, rest=0."""
    labels = torch.zeros_like(logits); labels[0] = 1.0
    return nn.MultiLabelSoftMarginLoss()(logits.unsqueeze(0), labels.unsqueeze(0))

def loss_kl_divergence(logits):
    """
    KL Divergence distillation: teacher = cosine-sim soft labels.
    Scores at idx 0 are gold, so give gold score=1.0, others = 0.1/N.
    """
    N = logits.shape[0]
    teacher = torch.full((N,), 0.1 / max(N-1, 1), device=logits.device)
    teacher[0] = 1.0
    teacher = teacher / teacher.sum()
    log_probs = F.log_softmax(logits, dim=0)
    return F.kl_div(log_probs, teacher, reduction='sum')

def loss_triplet_margin(logits, margin=0.5):
    """Triplet: gold (idx 0) vs hardest negative (max score among negs)."""
    gold_score = logits[0]
    if logits.shape[0] < 2: return torch.tensor(0.0, device=logits.device)
    hardest_neg = logits[1:].max()
    return F.relu(margin - gold_score + hardest_neg)

LOSS_FNS = {
    'InfoNCE':          loss_infonce,
    'FocalBCE':         loss_focal_bce,
    'SoftMargin':       loss_multilabel_soft_margin,
    'KL-Distill':       loss_kl_divergence,
    'Triplet-Margin':   loss_triplet_margin,
}

# ── Dataset ───────────────────────────────────────────────────────────────────
class SmallDataset(Dataset):
    def __init__(self, path, limit=500):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.data = [s for s in raw if any(c['is_gold'] for c in s['candidates'])][:limit]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ── Quick evaluator (Hit@1 on 200 dev samples from the training data) ─────────
def quick_eval(model, tok, samples, device, n=200):
    model.eval()
    hit = 0
    samples = [s for s in samples if any(c['is_gold'] for c in s['candidates'])][:n]
    with torch.no_grad():
        for item in samples:
            golds = [c for c in item['candidates'] if c['is_gold']]
            negs  = [c for c in item['candidates'] if not c['is_gold']]
            if not golds: continue
            cands = golds + negs[:19]  # top 20
            q = item['question']
            names = [c['name'] for c in cands]
            enc = tok([q]*len(cands), names, padding=True, truncation=True,
                      max_length=128, return_tensors='pt').to(device)
            with autocast():
                logits = model(**enc).logits.squeeze(-1)
            best = torch.argmax(logits).item()
            if cands[best]['is_gold']: hit += 1
    return hit / len(samples) * 100 if samples else 0.0

# ── Train one model with one loss function (2 epochs, 500 samples) ────────────
def train_one(loss_name, loss_fn, train_data, dev_data, device, epochs=2):
    tok   = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1).to(device)
    opt   = AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    loader = DataLoader(SmallDataset.__new__(SmallDataset), batch_size=8, collate_fn=lambda x: x)
    # Use train_data directly
    random.shuffle(train_data)
    
    print(f"\n  [{loss_name}] Training {epochs} epochs on {len(train_data)} samples...")
    for ep in range(epochs):
        model.train(); total = 0; count = 0
        random.shuffle(train_data)
        # Mini-batch manually
        for i in range(0, len(train_data), 8):
            batch = train_data[i:i+8]
            all_qs, all_es, offsets, offset = [], [], [], 0
            for item in batch:
                golds = [c for c in item['candidates'] if c['is_gold']]
                negs  = [c for c in item['candidates'] if not c['is_gold']]
                if not golds or not negs: continue
                cands = golds[:1] + random.sample(negs, min(15, len(negs)))
                N = len(cands)
                all_qs.extend([item['question']] * N)
                all_es.extend([c['name'] for c in cands])
                offsets.append((offset, offset + N))
                offset += N
            if not all_qs: continue
            enc = tok(all_qs, all_es, padding=True, truncation=True,
                      max_length=128, return_tensors='pt').to(device)
            with autocast():
                all_logits = model(**enc).logits.squeeze(-1)
                losses = [loss_fn(all_logits[s:e]) for s, e in offsets]
                loss = torch.stack(losses).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            total += loss.item(); count += 1
        print(f"    Epoch {ep+1}/{epochs} loss={total/max(count,1):.4f}")

    hit1 = quick_eval(model, tok, dev_data, device)
    print(f"  [{loss_name}] Hit@1 on 200 dev samples: {hit1:.2f}%")
    return hit1

# ── Main Comparison ───────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LossCompare] Device: {device}")

    # Use the existing 27k training data (already harvested)
    train_file = os.path.join(ROOT, 'data/exp16_cds_train_full.json')
    if not os.path.exists(train_file):
        print(f"[ERROR] {train_file} not found. Run harvest first.")
        return

    with open(train_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    valid = [s for s in raw if any(c['is_gold'] for c in s['candidates'])]
    random.shuffle(valid)
    train_data = valid[:500]
    dev_data   = valid[500:700]   # 200 held-out for quick eval
    print(f"[LossCompare] Train: {len(train_data)} | Dev: {len(dev_data)}")

    results = {}
    for name, fn in LOSS_FNS.items():
        try:
            hit1 = train_one(name, fn, train_data, dev_data, device)
            results[name] = hit1
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results[name] = -1.0

    print("\n" + "="*55)
    print("LOSS FUNCTION COMPARISON — Stage 3 (Cross-Encoder)")
    print("="*55)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← WINNER" if score == max(results.values()) else ""
        print(f"  {name:<22} Hit@1 = {score:.2f}%{marker}")
    print("="*55)

    winner = max(results, key=lambda k: results[k])
    print(f"\n[LossCompare] WINNER: {winner} ({results[winner]:.2f}%)")
    print(f"[LossCompare] Update exp16v2_train.py Stage 3 loss to: {winner}")

if __name__ == "__main__":
    main()
