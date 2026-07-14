"""
train_planner.py — Stage 1: Unified Adaptive Cross-Hop Coherence Planning (Exp 6)
==================================================================================

PURPOSE
-------
This is the **Stage 1 Planner** in the two-stage KGQA pipeline.  It trains a
unified model that jointly solves two sub-problems:

  1. **Domain classification** (Progressive Constraint Tightening, Exp 3):
     Given a natural-language question, predict the top-level Freebase domain
     (e.g. "music", "film") that the answer entity belongs to.  This acts as a
     coarse, progressive constraint that narrows the relation search space before
     any hop-by-hop traversal begins.

  2. **Cross-hop relation planning** (Cross-Hop Coherence Planning, Exp 4):
     For each hop in a multi-hop reasoning path (up to `max_hops` steps),
     predict (a) the correct Freebase relation to follow and (b) whether
     traversal should stop at that hop.  A Transformer encoder over learned
     hop-position embeddings enforces *coherence* across hops — the model sees
     all hops simultaneously rather than greedily one at a time.

PAPER SECTION
-------------
Corresponds to **Experiment 6** ("Unified Adaptive-CHCP") in the paper, which
combines the progressive-constraint idea of Exp 3 with the cross-hop coherence
architecture of Exp 4 into a single jointly-trained model.

PIPELINE POSITION
-----------------
  [CWQ JSON] → UnifiedDataset (parses SPARQL paths)
             → UnifiedKGQAPlanner (BERT + Transformer)
             → three losses (domain CE + relation CE + stop BCE)
             → checkpoint: checkpoints/exp6_unified_best.pt
             ↓
  [Stage 2 — RL agents (train_rlmc.py / train_strl.py) load exp7_roberta_best.pt
   which is a scaled-up version trained from this Exp 6 design]

INPUTS
------
  data/cwq_train.json                   : CWQ training split (question + SPARQL)
  data/cwq_dev.json                     : CWQ dev split
  data/processed_entity/relation2id.pt  : dict[rel_str -> int]
  data/processed_entity/domain2id.pt    : dict[domain_str -> int]

OUTPUTS
-------
  checkpoints/exp6_unified_best.pt : best model weights (saved when dev
                                     relation loss improves)
  metrics/exp6_unified.csv         : per-epoch dev relation loss

KEY HYPERPARAMETERS
-------------------
  hidden_dim   = 256        (BERT projection size; Exp 7 scales this to 512)
  max_hops     = 4          (maximum reasoning depth)
  nhead        = 4          (Transformer multi-head attention heads)
  num_layers   = 2          (Transformer encoder depth)
  batch_size   = 16 (train), 32 (dev)
  lr           = 2e-5       (AdamW)
  epochs       = 30
  loss weights = 1 : 1 : 1  (domain : relation : stop)

HOW IT WORKS
------------
  1. BERT encodes the question -> CLS token -> linear projection -> h_q [B, 256]
  2. Domain head: h_q -> cross-entropy over domain classes  (coarse constraint)
  3. Confidence head: h_q -> sigmoid scalar  (not used in loss; available at
     inference for adaptive constraint tightening)
  4. Hop embeddings: learned position vectors [max_hops, 256] are added to h_q
     and fed through a 2-layer Transformer -> refined_repr [B, max_hops, 256]
  5. Relation head: refined_repr -> cross-entropy over all relations per hop
  6. Stop head: refined_repr -> binary cross-entropy (1=valid hop, 0=padding)
  7. Total loss = L_domain + L_relation + L_stop  (equal weights)
  8. Mixed-precision training via torch.amp; best checkpoint saved on dev L_rel.
"""

import os, sys, json, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm

# ── Resolve project root so shared modules can be imported regardless of CWD ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared.encoder import QuestionEncoder
from utils.sparql_parser import find_reasoning_path

# ──────────────────────────────────────────────────────────────────────────────
#  Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedKGQAPlanner(nn.Module):
    """
    Unified planner that combines:
      * Domain / confidence heads   (Exp 3 — Progressive Constraint Tightening)
      * Cross-hop Transformer heads (Exp 4 — Cross-Hop Coherence Planning)

    All heads share the same BERT question encoder and hidden projection, so
    the domain constraint and the per-hop relation predictions are informed by
    the same question representation.
    """
    def __init__(self, num_domains, num_relations, hidden_dim=256, max_hops=4):
        super().__init__()
        self.max_hops = max_hops

        # ── Shared question encoder (BERT-base, fine-tuned end-to-end) ────────
        self.q_encoder = QuestionEncoder(model_name="bert-base-uncased")
        # Project BERT's output dim (768) down to hidden_dim for efficiency
        self.proj = nn.Linear(self.q_encoder.output_dim, hidden_dim)

        # 1. Progressive Constraint heads (from Exp 3)
        # domain_head produces logits over top-level Freebase domains
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        # confidence_head produces a scalar in [0,1]; high confidence -> use
        # TIGHT beam width at inference, low -> fall back to LOOSE
        self.confidence_head = nn.Linear(hidden_dim, 1) # scalar confidence

        # 2. Coherent Planner (from Exp 4)
        # Learnable positional embeddings for each hop slot; added to h_q so
        # the Transformer can attend across hops with positional context.
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        # 2-layer Transformer encoder refines hop-augmented representations so
        # each hop's prediction is conditioned on all other hops simultaneously.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Per-hop relation scorer over the full relation vocabulary
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        # Per-hop binary stop signal: positive -> valid hop, negative -> padding
        self.adaptive_stop_head = nn.Linear(hidden_dim, 1) # learned stop per hop

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids      : [B, seq_len]  BERT token IDs
            attention_mask : [B, seq_len]  1=real token, 0=padding

        Returns (dict):
            domain_logits : [B, num_domains]             raw domain scores
            confidence    : [B, 1]                       sigmoid confidence in (0,1)
            rel_logits    : [B, max_hops, num_relations] per-hop relation scores
            stop_logits   : [B, max_hops]                per-hop stop scores (pre-sigmoid)
        """
        B = input_ids.size(0)

        # ── Step 1: Encode question with BERT, take CLS representation ────────
        # BERT Encoding
        q_h = self.q_encoder(input_ids, attention_mask)  # [B, bert_dim]
        h_q = self.proj(q_h)   # [B, hidden_dim] — compressed question embedding

        # ── Step 2: Coarse domain prediction (Exp 3 elements) ─────────────────
        domain_logits = self.domain_head(h_q)                        # [B, num_domains]
        q_confidence  = torch.sigmoid(self.confidence_head(h_q))     # [B, 1] in (0,1)

        # ── Step 3: Cross-Hop Reasoning (Exp 4 elements) ──────────────────────
        # Combine question with learned hop positions:
        # Broadcast h_q over max_hops positions and add learnable hop embeddings.
        # Shape: [B, 1, H] + [1, max_hops, H] -> [B, max_hops, H]
        init_repr    = h_q.unsqueeze(1) + self.hop_embeddings.unsqueeze(0) # [B, max_hops, hidden_dim]
        # Self-attention across all hops so each hop can inform the others
        refined_repr = self.transformer(init_repr) # [B, max_hops, hidden_dim]

        # ── Step 4: Per-hop output heads ──────────────────────────────────────
        rel_logits  = self.relation_head(refined_repr) # [B, max_hops, num_relations]
        stop_logits = self.adaptive_stop_head(refined_repr).squeeze(-1) # [B, max_hops]

        return {
            'domain_logits': domain_logits,
            'confidence':    q_confidence,
            'rel_logits':    rel_logits,
            'stop_logits':   stop_logits,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset and Collate
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedDataset(Dataset):
    """
    Reads the CWQ JSON file and produces training samples with:
      question  : raw question string (tokenised lazily in collate_fn)
      domain    : integer domain ID of the first relation in the path
      path      : list of relation IDs, length max_hops, right-padded with 0
      num_hops  : number of valid (non-padded) hops

    SPARQL parsing extracts the multi-hop reasoning path as a list of
    (subject, relation, object, direction) tuples via `find_reasoning_path`.
    Samples where any relation is out-of-vocabulary are silently skipped.
    """
    def __init__(self, data_path, relation2id, domain2id, max_hops=4):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = []
        for item in data:
            # Parse the SPARQL into an ordered relation path
            path = find_reasoning_path(item['sparql'])
            if path is None: continue  # skip questions with unparseable SPARQL

            # ── Derive domain from first relation's Freebase prefix ───────────
            # e.g. "music.concert_tour.artist" -> domain = "music"
            # Domain from first relation path
            main_rel = path[0][1]
            domain   = main_rel.split('.')[0] if '.' in main_rel else 'none'
            if domain not in domain2id: domain = 'none'  # fall back to catch-all

            # ── Build relation ID sequence, skip OOV samples ──────────────────
            # Relation IDs for each hop
            rel_ids = []
            valid   = True
            for _, rel, _, _ in path:
                if rel in relation2id:
                    rel_ids.append(relation2id[rel])
                else:
                    valid = False; break   # any OOV relation -> discard sample

            if valid:
                # pad/truncate to exactly max_hops slots
                num_hops = len(rel_ids)
                if num_hops > max_hops: rel_ids = rel_ids[:max_hops]
                else:                   rel_ids = rel_ids + [0]*(max_hops - num_hops)

                self.samples.append({
                    'question': item['question'],
                    'domain':   domain2id[domain],
                    'path':     rel_ids,
                    'num_hops': min(num_hops, max_hops),
                })

    def __len__(self):           return len(self.samples)
    def __getitem__(self, idx):  return self.samples[idx]


def collate_unified(batch, tokenizer):
    """
    Custom collate function passed to DataLoader.
    Tokenises raw question strings on-the-fly, returning padded BERT tensors
    alongside pre-encoded label tensors.

    Args:
        batch     : list of dicts from UnifiedDataset.__getitem__
        tokenizer : BertTokenizer instance, bound via functools.partial

    Returns:
        encoded  : BatchEncoding with 'input_ids' and 'attention_mask'  [B, seq]
        domains  : [B]              gold domain indices
        paths    : [B, max_hops]   gold relation indices (padded with 0)
        nums     : [B]             number of valid hops per sample
    """
    questions = [s['question'] for s in batch]
    domains   = torch.tensor([s['domain']   for s in batch])
    paths     = torch.tensor([s['path']     for s in batch])
    nums      = torch.tensor([s['num_hops'] for s in batch])

    # BERT tokenisation with dynamic padding to the longest sequence in the batch
    encoded = tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors='pt')
    return encoded, domains, paths, nums


# ──────────────────────────────────────────────────────────────────────────────
#  Training Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def train_unified():
    """
    Full training loop for the Unified Adaptive-CHCP planner (Exp 6).

    Three losses are computed jointly every batch:
      L_domain : cross-entropy over predicted domain vs. gold domain
      L_rel    : cross-entropy over predicted relation vs. gold relation,
                 computed across ALL hops at once (logits flattened to [B*H, R])
      L_stop   : binary cross-entropy; target = 1 for valid hops, 0 for padding

    Dev evaluation reports only L_rel (the most informative single metric for
    downstream hop accuracy); the best checkpoint is saved when L_rel improves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load pre-built vocabulary maps ────────────────────────────────────────
    # Load Maps
    rel2id  = torch.load('data/processed_entity/relation2id.pt')
    dom2id  = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id)
    num_dom = len(dom2id)

    # ── Instantiate model, optimiser, tokeniser ───────────────────────────────
    # Setup Model
    model     = UnifiedKGQAPlanner(num_dom, num_rel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ── Build datasets and loaders ────────────────────────────────────────────
    # Dataset
    train_ds = UnifiedDataset('data/cwq_train.json', rel2id, dom2id)
    dev_ds   = UnifiedDataset('data/cwq_dev.json',   rel2id, dom2id)
    collate  = functools.partial(collate_unified, tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=32,               collate_fn=collate)

    # ── Training config ───────────────────────────────────────────────────────
    # Loop
    epochs    = 30
    best_loss = float('inf')
    # GradScaler enables AMP (FP16) on CUDA to reduce memory and speed up
    # BERT forward passes; a no-op when running on CPU.
    scaler = torch.amp.GradScaler('cuda')

    # ── Metrics file initialisation ───────────────────────────────────────────
    metrics_dir  = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp6_unified.csv")
    with open(metrics_path, "w") as f:
        f.write("epoch,dev_loss\n")

    print("\nTraining Unified Adaptive-CHCP (Exp 6)...")
    for epoch in range(epochs):
        model.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        # ── Per-batch train loop ──────────────────────────────────────────────
        for enc, doms, paths, nums in t_bar:
            enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)

            with torch.amp.autocast('cuda'):
                out = model(enc['input_ids'], enc['attention_mask'])

                # ── Loss 1: Domain Loss ────────────────────────────────────────
                # Standard cross-entropy; doms = [B] gold domain index
                loss_dom = F.cross_entropy(out['domain_logits'], doms)

                # ── Loss 2: Relation Planning Loss ─────────────────────────────
                # paths: [B, max_hops]
                # Flatten [B, max_hops, num_rel] -> [B*max_hops, num_rel] so CE
                # operates over each (sample, hop) pair independently.
                loss_rel = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))

                # ── Loss 3: Stop Loss ──────────────────────────────────────────
                # Binary target for each hop: 1 if hop is valid, 0 if it's padding
                # nums: [B]
                # Build a binary target: for sample b, hops 0..num_hops-1 are 1.0
                B, H = paths.size()
                stop_targets = torch.zeros(B, H).to(device)
                for b in range(B):
                    stop_targets[b, :nums[b]] = 1.0

                loss_stop = F.binary_cross_entropy_with_logits(out['stop_logits'], stop_targets)

                # ── Combined loss (equal weights) ──────────────────────────────
                total_loss = loss_dom + loss_rel + loss_stop

            # AMP-aware backward pass + gradient update
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_bar.set_postfix(loss=total_loss.item())

        # ── Dev evaluation (relation loss only as primary metric) ─────────────
        # Eval
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for enc, doms, paths, nums in dev_loader:
                enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)
                out  = model(enc['input_ids'], enc['attention_mask'])
                # Use relation CE as the primary dev metric
                loss = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                v_loss += loss.item()

        avg_v = v_loss / len(dev_loader)
        print(f"Epoch {epoch} | Dev Rel Loss: {avg_v:.4f}")

        # Append epoch metrics to CSV
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_v:.4f}\n")

        # ── Checkpoint on improvement ─────────────────────────────────────────
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), 'checkpoints/exp6_unified_best.pt')


if __name__ == "__main__":
    train_unified()
