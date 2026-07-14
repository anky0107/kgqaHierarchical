"""
Exp 19: Learned Relation Embedding Bank for Stage 2 Path-Aware Ranker
======================================================================

HYPOTHESIS
----------
Stage 2 (exp16v2) encodes path strings like:
    "film.film.directed_by people.person.nationality"
using all-mpnet-base-v2, which was pretrained on natural-language sentences.
It produces near-random embeddings for Freebase relation IDs because it has
never seen this vocabulary.

Fix: replace the MPNet path encoder with a compact learned relation embedding
bank — one trainable vector per unique relation in the vocabulary.

The path representation h_p is computed as:
    h_p = mean( E[r_1], E[r_2], ..., E[r_k] )
where E ∈ R^{|R| × emb_dim} is a learnable embedding table initialised from
the frozen RoBERTa-large relation-text embeddings (same as exp15's
RelationEmbeddingBank), then fine-tuned jointly with the ranker.

The rest of the Stage 2 PathAwareRanker (MLP fusion head, SoftMargin loss,
question encoder) is unchanged from exp16v2.

PATH FORMAT IN JSON
-------------------
Path strings in the CDS dataset may be:
  (a) space-separated relation ID strings: "film.film.directed_by people.person.nationality"
  (b) pipe-separated: "film.film.directed_by|people.person.nationality"
  (c) space-separated integer IDs: "42 187"
All three are handled by parse_path_to_ids().

EXPECTED GAIN: +2–4% Hit@1 at Stage 2 output
CHECKPOINT:    checkpoints/exp19_s2_relembbank.pt
METRICS:       metrics/exp19_s2_relembbank.csv
"""

import os, sys, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()

# ─────────────────────────────────────────────────────────────
#  Relation embedding bank init from frozen RoBERTa text reps
# ─────────────────────────────────────────────────────────────

def _rel_id_to_text(rel_id: str) -> str:
    """Same heuristic as exp15_strl.py RelationEmbeddingBank._rel_to_text."""
    parts = rel_id.split(".")
    if len(parts) >= 3:
        subject   = parts[-2].replace("_", " ")
        predicate = parts[-1].replace("_", " ")
        if (predicate.endswith("s")
                or "owned"   in predicate
                or "founded" in predicate
                or "won"     in predicate):
            return f"{subject} HAS {predicate}"
        return f"{subject} {predicate}"
    if len(parts) == 2:
        return parts[-1].replace("_", " ")
    return rel_id.replace(".", " ").replace("_", " ")


def build_rel_emb_init(id2rel: dict, emb_dim: int,
                        device: torch.device,
                        roberta_batch: int = 64) -> torch.Tensor:
    """
    Pre-compute [|R|, emb_dim] init matrix from frozen RoBERTa-large.
    Result is projected to emb_dim with PCA if emb_dim < 1024,
    or a linear layer if emb_dim > 1024.
    Used only once; result is saved to cache.
    """
    cache_path = os.path.join(
        ROOT, f"data/processed_entity/rel_emb_init_{emb_dim}d.pt")

    if os.path.exists(cache_path):
        print(f"[Exp19] Loading rel-emb init from cache: {cache_path}")
        return torch.load(cache_path, map_location=device)

    print(f"[Exp19] Building rel-emb init ({len(id2rel)} rels, "
          f"emb_dim={emb_dim}) …")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    encoder   = RobertaModel.from_pretrained("roberta-large").to(device)
    encoder.eval()

    N = len(id2rel)
    texts = [_rel_id_to_text(id2rel[i]) for i in range(N)]

    all_embs = []
    with torch.no_grad():
        for start in range(0, N, roberta_batch):
            batch_texts = texts[start: start + roberta_batch]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=32, return_tensors="pt").to(device)
            cls = encoder(**enc).last_hidden_state[:, 0, :]   # [B, 1024]
            all_embs.append(cls.cpu())

    embs = torch.cat(all_embs, dim=0)   # [N, 1024]

    if emb_dim == 1024:
        init = embs
    elif emb_dim < 1024:
        # PCA projection: take top-emb_dim principal components
        U, S, Vt = torch.pca_lowrank(embs, q=emb_dim, niter=4)
        init = embs @ Vt                  # [N, emb_dim]
    else:
        proj = nn.Linear(1024, emb_dim, bias=False)
        nn.init.orthogonal_(proj.weight)
        with torch.no_grad():
            init = proj(embs)             # [N, emb_dim]

    del encoder
    torch.cuda.empty_cache()

    torch.save(init.cpu(), cache_path)
    print(f"[Exp19] Rel-emb init saved → {cache_path}  shape={init.shape}")
    return init.to(device)


# ─────────────────────────────────────────────────────────────
#  Path parsing  (handles all three formats)
# ─────────────────────────────────────────────────────────────

def parse_path_to_ids(path_str: str, rel2id: dict) -> list:
    """
    Convert a stored path string to a list of integer relation IDs.
    Returns [] if the path is empty or unknown.
    """
    if not path_str or not path_str.strip():
        return []

    sep    = "|" if "|" in path_str else " "
    tokens = [t.strip() for t in path_str.split(sep) if t.strip()]

    ids = []
    for tok in tokens:
        if tok.isdigit():
            ids.append(int(tok))
        elif tok in rel2id:
            ids.append(rel2id[tok])
        # unknown relation → skip (keeps list shorter but valid)
    return ids


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class RelEmbPathRanker(nn.Module):
    """
    Stage 2 ranker with a learned relation embedding bank for path encoding.

    Architecture:
      - question / entity: encoded by all-mpnet-base-v2 (unchanged from exp16v2)
      - path h_p         : mean of trainable relation embeddings (NEW)
      - fusion           : MLP([e_q ; h_p ; h_e]) → scalar score (unchanged)
    """
    def __init__(self, rel_emb_init: torch.Tensor,
                 sbert_model: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        num_rels, emb_dim = rel_emb_init.shape

        # ── Relation embedding table (fine-tuned) ─────────────────────────
        self.rel_emb = nn.Embedding(num_rels, emb_dim)
        with torch.no_grad():
            self.rel_emb.weight.copy_(rel_emb_init)

        # ── Question / entity encoder ──────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(sbert_model)
        text_dim     = self.encoder.config.hidden_size   # 768 for mpnet

        # ── Projection to align path and text dims ─────────────────────────
        # If emb_dim ≠ text_dim we project path rep to text_dim
        self.path_proj = (nn.Linear(emb_dim, text_dim, bias=False)
                          if emb_dim != text_dim else nn.Identity())

        # ── Fusion MLP (same as exp16v2) ───────────────────────────────────
        self.fuse = nn.Sequential(
            nn.Linear(text_dim * 3, text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim, 1)
        )

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask=attention_mask) \
                   .last_hidden_state[:, 0, :]

    def encode_path(self, rel_id_lists: list) -> torch.Tensor:
        """
        rel_id_lists: list of N lists, each containing int relation IDs.
        Returns [N, text_dim] mean-pooled path representations.
        Zero vector for empty paths.
        """
        device = self.rel_emb.weight.device
        reps = []
        for ids in rel_id_lists:
            if ids:
                idx  = torch.tensor(ids, dtype=torch.long, device=device)
                mean = self.rel_emb(idx).mean(dim=0)        # [emb_dim]
            else:
                mean = torch.zeros(
                    self.rel_emb.embedding_dim, device=device)
            reps.append(mean)
        h_p = torch.stack(reps, dim=0)                      # [N, emb_dim]
        return self.path_proj(h_p)                           # [N, text_dim]

    def forward(self, q_ids, q_mask, rel_id_lists, e_ids, e_mask):
        """
        q_ids, q_mask   : tokenised questions  [N, seq_len]
        rel_id_lists    : list of N lists of int rel IDs
        e_ids, e_mask   : tokenised entity names [N, seq_len]
        Returns         : [N] scores
        """
        e_q = self.encode_text(q_ids, q_mask)       # [N, text_dim]
        h_p = self.encode_path(rel_id_lists)         # [N, text_dim]
        h_e = self.encode_text(e_ids, e_mask)        # [N, text_dim]
        return self.fuse(torch.cat([e_q, h_p, h_e], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Loss  (same as exp16v2 Stage 2)
# ─────────────────────────────────────────────────────────────

def loss_soft_margin(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(logits); labels[0] = 1.0
    return nn.MultiLabelSoftMarginLoss()(
        logits.unsqueeze(0), labels.unsqueeze(0))


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class RelEmbCDSDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, rel2id: dict, max_neg: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.rel2id  = rel2id
        self.max_neg = max_neg
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp19 Dataset] {len(self.samples)} samples  "
              f"from {os.path.basename(json_path)}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

    def get_path_ids(self, candidate: dict, item_path: str) -> list:
        path_str = candidate.get("path") or item_path or ""
        return parse_path_to_ids(path_str, self.rel2id)


def collate_passthrough(batch):
    return batch


# ─────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────

class Stage2RelEmbTrainer:
    def __init__(self, device: torch.device, rel_emb_init: torch.Tensor,
                 sbert_model: str = "sentence-transformers/all-mpnet-base-v2",
                 lr: float = 5e-5, max_neg: int = 15):
        self.device  = device
        self.max_neg = max_neg
        self.tok     = AutoTokenizer.from_pretrained(sbert_model)
        self.model   = RelEmbPathRanker(rel_emb_init, sbert_model).to(device)
        # Higher LR for embedding table, standard LR for transformer
        self.opt     = AdamW([
            {"params": self.model.rel_emb.parameters(),  "lr": lr * 5},
            {"params": self.model.path_proj.parameters(), "lr": lr * 2},
            {"params": self.model.fuse.parameters(),      "lr": lr},
            {"params": self.model.encoder.parameters(),   "lr": lr},
        ], weight_decay=1e-2)
        print("[Exp19] Model ready. Param groups: "
              "rel_emb (5×lr), path_proj (2×lr), fuse (1×lr), encoder (1×lr)")

    def train(self, dataset: RelEmbCDSDataset, loader: DataLoader,
              epochs: int = 5, accum_steps: int = 4):
        metrics_dir  = os.path.join(ROOT, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "exp19_s2_relembbank.csv")
        with open(metrics_path, "w") as f:
            f.write("epoch,avg_loss\n")

        scaler = GradScaler("cuda")
        print(f"\n[Exp19 S2] Training  |  epochs={epochs}  accum={accum_steps}")

        for ep in range(epochs):
            self.model.train()
            total = 0.0; count = 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                all_q, all_e, all_path_ids = [], [], []
                offsets = []; offset = 0

                for item in batch:
                    q         = str(item["question"])
                    item_path = item.get("path") or ""
                    golds = [c for c in item["candidates"] if     c["is_gold"]]
                    negs  = [c for c in item["candidates"] if not c["is_gold"]]
                    if not golds or not negs:
                        continue

                    cands = golds[:1] + random.sample(negs, min(self.max_neg, len(negs)))
                    N = len(cands)

                    for c in cands:
                        all_q.append(q)
                        all_e.append(str(c.get("name", "")))
                        all_path_ids.append(dataset.get_path_ids(c, item_path))

                    offsets.append((offset, offset + N))
                    offset += N

                if not all_q:
                    continue

                qe = self.tok(all_q, padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(self.device)
                ee = self.tok(all_e, padding=True, truncation=True,
                              max_length=64, return_tensors="pt").to(self.device)

                with autocast("cuda"):
                    scores = self.model(
                        qe["input_ids"], qe["attention_mask"],
                        all_path_ids,
                        ee["input_ids"], ee["attention_mask"])
                    loss = torch.stack([
                        loss_soft_margin(scores[s:e]) for s, e in offsets
                    ]).mean() / accum_steps

                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(self.opt); scaler.update(); self.opt.zero_grad()

                total += loss.item() * accum_steps; count += 1
                pbar.set_postfix(loss=f"{total/count:.4f}")

            avg = total / max(count, 1)
            print(f"  Ep{ep+1} avg_loss={avg:.4f}")
            with open(metrics_path, "a") as f:
                f.write(f"{ep+1},{avg:.4f}\n")

        ckpt = os.path.join(ROOT, "checkpoints/exp19_s2_relembbank.pt")
        torch.save(self.model.state_dict(), ckpt)
        print(f"[Exp19] Checkpoint → {ckpt}")


# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, dataset: RelEmbCDSDataset,
             device: torch.device) -> float:
    model.eval()
    hits = 0; total = 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc="Evaluating"):
        item      = batch[0]
        q         = str(item["question"])
        item_path = item.get("path") or ""
        cands     = item["candidates"]
        if not cands: continue
        gold_idx  = next((i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None: continue

        all_q   = [q] * len(cands)
        all_e   = [str(c.get("name", "")) for c in cands]
        all_ids = [dataset.get_path_ids(c, item_path) for c in cands]

        qe = tok(all_q, padding=True, truncation=True,
                 max_length=128, return_tensors="pt").to(device)
        ee = tok(all_e, padding=True, truncation=True,
                 max_length=64,  return_tensors="pt").to(device)

        scores  = model(qe["input_ids"], qe["attention_mask"],
                        all_ids, ee["input_ids"], ee["attention_mask"])
        if torch.argmax(scores).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"[Exp19 Eval] Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp19] Device: {device}")

    rel2id = torch.load(
        os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    id2rel = {v: k for k, v in rel2id.items()}

    # emb_dim=256 gives a compact 256-d relation table;
    # raise to 512 or 768 if you have VRAM to spare.
    EMB_DIM = 256
    rel_emb_init = build_rel_emb_init(id2rel, EMB_DIM, device)

    train_file = os.path.join(ROOT, "data/exp16_cds_train.json")
    dev_file   = os.path.join(ROOT, "data/exp16_cds_dev.json")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")

    train_ds     = RelEmbCDSDataset(train_file, rel2id)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=collate_passthrough, pin_memory=True)

    trainer = Stage2RelEmbTrainer(device, rel_emb_init)
    trainer.train(train_ds, train_loader, epochs=5, accum_steps=4)

    if os.path.exists(dev_file):
        dev_ds = RelEmbCDSDataset(dev_file, rel2id)
        evaluate(trainer.model, trainer.tok, dev_ds, device)
    else:
        print(f"[Exp19] Dev file not found at {dev_file} — skipping eval.")


if __name__ == "__main__":
    main()
