"""
train_f2_path_ranker.py — CDS Pipeline: F2 MPNet Path-Aware Ranker (Exp-27)
=============================================================================

Paper Section: §V-D  "Cascading Dual-Stage Filtering (CDS) — F2 Path-Aware Ranker"

Purpose
-------
Trains the second-stage (F2) ranker of the CDS pipeline.  F2 receives the
top-200 candidates produced by the MiniLM-L6-v2 bi-encoder (F1) and re-ranks
them, retaining the top-50 for the downstream Flan-T5 generative judge (F3).

This file implements **Exp-27**, which upgrades the original binary
soft-margin loss (Exp-18) to a **listwise KL-Distillation loss** that trains
F2 as an explicit ranker over the full candidate set rather than as an
independent binary classifier.  The insight (confirmed in §V-E ablation) is
that ranking 200→50 is a fundamentally different task from point-wise
classification: listwise training aligns training objective with inference
objective, reducing gold-entity recall loss at the F2 cut.

Architecture: PathAwareRanker  (§V-D, Fig. 3)
----------------------------------------------
  • Shared encoder : sentence-transformers/all-mpnet-base-v2  (hidden=768)
  • Three independent CLS embeddings are extracted for:
        q  — natural-language question
        p  — relation-path context (hop relations concatenated as a string)
        e  — candidate entity name
  • Fusion head   : MLP([q_emb ⊕ p_emb ⊕ e_emb])  →  scalar score
        Linear(768×3 → 768) → GELU → Dropout(0.1) → Linear(768 → 1)
  • Score = MLP([q; p; e]).squeeze()

Training Loss: KL-Distillation Listwise (loss_kl_distill)
----------------------------------------------------------
  Given a set of N candidates for one question (1 gold + ≤15 hard negatives):
    teacher[0]   = 1.0           (gold entity gets full probability mass)
    teacher[1:N] = 0.1 / (N-1)  (negatives share a small residual probability)
    teacher      = teacher / sum(teacher)   (normalise to a valid distribution)
    loss         = KL(log_softmax(model_scores) ‖ teacher)
  This soft target prevents overconfident gradient signals on hard examples and
  provides richer gradient information than a one-hot label — a technique
  borrowed from knowledge distillation literature.

Pipeline position
-----------------
  [F1] top-200 candidates
        │
        ▼
  [train_f2_path_ranker.py]  ← THIS FILE
        │  checkpoint: checkpoints/exp27_s2_mpnet.pt
        ▼
  [F3] top-50 passed to Flan-T5

Inputs
------
- data/exp26_s2_hard_negatives.json  : training set with hard negatives (27k)
- data/exp16_cds_dev.json            : dev set for isolated Hit@1 evaluation
  Each record: {"question": str, "path": ..., "candidates": [{mid, name, is_gold}]}

Outputs
-------
- checkpoints/exp27_s2_epoch{n}.pt   : per-epoch checkpoints
- checkpoints/exp27_s2_mpnet.pt      : final model weights
- metrics/exp27_s2_mpnet.csv         : epoch-level avg loss log

Key hyperparameters
-------------------
- lr            : 5e-5 (AdamW, weight_decay=1e-2)
- batch_size    : 4  (effective = 4 × accum_steps=4 = 16)
- max_neg       : 15 hard negatives per question
- epochs        : 5
- max_length_q  : 128 tokens  (question)
- max_length_p  : 64  tokens  (path string)
- max_length_e  : 64  tokens  (entity name)
- Mixed precision: torch.amp autocast (float16 on CUDA)

Exp-25 → Exp-27 lineage
------------------------
Exp-25 introduced the listwise loss concept for Stage-2.  Exp-27 is the
production version with the MPNet encoder substituted for the MiniLM used in
Exp-25, producing the checkpoint evaluated in the paper's ablation table.
"""

import os, sys, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

# ── Global constants ──────────────────────────────────────────────────────────
# Using MPNet (all-mpnet-base-v2) as the shared encoder for F2.
# MPNet's permutation-based pre-training gives better cross-sentence
# representations than MiniLM — critical when comparing question vs path vs
# entity in a shared embedding space.
ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
CKPT_NAME    = "exp27_s2_mpnet.pt"
METRICS_CSV  = "exp27_s2_mpnet.csv"
TRAIN_FILE   = "data/exp26_s2_hard_negatives.json"
DEV_FILE     = "data/exp16_cds_dev.json"


# ──────────────────────────────────────────────────────────────────────────────
# PathAwareRanker  (canonical copy from cds_pipeline/models.py)
# ──────────────────────────────────────────────────────────────────────────────

class PathAwareRanker(nn.Module):
    """
    Stage 2 (F2) ranking model: MPNet-base-v2 shared encoder + 3-input MLP fusion head.

    Architecture (§V-D, Fig. 3):
        Encoder  : all-mpnet-base-v2  →  CLS token  (hidden_size = 768)
        q_emb    : CLS(question)
        p_emb    : CLS(path string — hop relations concatenated)
        e_emb    : CLS(entity name)
        score    : MLP([q_emb ⊕ p_emb ⊕ e_emb])  →  scalar

    The SAME encoder weights process all three input types, enabling cross-type
    attention during fine-tuning and reducing parameter count relative to three
    separate encoders.

    Score = MLP([q_emb; p_emb; e_emb])  →  scalar
    """
    def __init__(self) -> None:
        super().__init__()
        # Shared MPNet encoder — all three input types pass through this
        self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
        hidden = self.encoder.config.hidden_size          # 768 for mpnet-base
        # Fusion MLP: concatenated triple → scalar relevance score
        #   Layer 1: 768*3 → 768  (compress the three CLS vectors)
        #   Activation: GELU (smoother than ReLU for ranking tasks)
        #   Dropout: 0.1  (light regularisation)
        #   Layer 2: 768 → 1  (scalar score)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self,
                q_ids:  torch.Tensor, q_mask: torch.Tensor,
                p_ids:  torch.Tensor, p_mask: torch.Tensor,
                e_ids:  torch.Tensor, e_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode question, path, and entity, then fuse to a score.

        Parameters
        ----------
        q_ids / q_mask  : tokenised question         [B, L_q]
        p_ids / p_mask  : tokenised path string      [B, L_p]
        e_ids / e_mask  : tokenised entity name      [B, L_e]

        Returns
        -------
        torch.Tensor of shape [B]  —  unbounded scalar scores (higher = more relevant)
        """
        enc = self.encoder
        # Extract CLS (index 0) from last hidden state for each input type
        q = enc(q_ids,  attention_mask=q_mask).last_hidden_state[:, 0, :]   # [B, 768]
        p = enc(p_ids,  attention_mask=p_mask).last_hidden_state[:, 0, :]   # [B, 768]
        e = enc(e_ids,  attention_mask=e_mask).last_hidden_state[:, 0, :]   # [B, 768]
        # Concatenate along feature dim → [B, 768*3], then score → [B]
        return self.fuse(torch.cat([q, p, e], dim=-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# KL-Distillation listwise loss
# ──────────────────────────────────────────────────────────────────────────────

def loss_kl_distill(scores: torch.Tensor) -> torch.Tensor:
    """
    Compute the listwise KL-Distillation loss for one question's candidate set.

    Assumption: the gold entity is ALWAYS at index 0 in `scores` (enforced by
    the collation logic which prepends `golds[:1]` before sampling negatives).

    Soft teacher distribution:
        teacher[0]   = 1.0          ← gold entity
        teacher[1:N] = 0.1/(N-1)    ← spread small residual mass over negatives
        teacher      = teacher / teacher.sum()   ← normalise

    Using soft labels (0.1 residual instead of hard 0) avoids saturating the
    softmax on hard negatives and provides smoother gradient flow — analogous
    to label smoothing in classification but adapted for listwise ranking.

    Loss = KL( log_softmax(scores) ‖ teacher )
         = Σ_i  teacher[i] * (log teacher[i] - log softmax(scores)[i])

    Parameters
    ----------
    scores : torch.Tensor of shape [N]  —  model scores for N candidates

    Returns
    -------
    torch.Tensor scalar  —  KL divergence (>=0; 0 means perfect ranking)
    """
    N = scores.shape[0]
    # Initialise all positions to the small negative residual mass
    teacher = torch.full((N,), 0.1 / max(N - 1, 1), device=scores.device)
    teacher[0] = 1.0          # gold entity at index 0 gets full probability
    teacher = teacher / teacher.sum()   # normalise to a valid distribution
    # KL divergence: F.kl_div expects (log_probs, target_probs)
    return F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction="sum")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ListwiseS2Dataset(Dataset):
    """
    PyTorch Dataset wrapping the hard-negative mining output from Exp-26/18.

    Each sample is a dict with keys: question, path, candidates.
    Only samples that contain at least one gold-labelled candidate are retained
    (samples without gold entities cannot produce a meaningful training signal).

    Parameters
    ----------
    json_path : str   — path to the candidate JSON file
    max_neg   : int   — maximum hard negatives to sample per question at
                        collation time (enforced in the trainer, not here)
    """
    def __init__(self, json_path: str, max_neg: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_neg = max_neg
        # Filter out samples with no gold candidate — they carry no supervision
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp27 Dataset] {len(self.samples)} samples with gold labels "
              f"from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_passthrough(batch):
    """
    Identity collate function — returns the raw list of dicts.

    Tokenisation is deferred to the trainer so that variable-length path/entity
    strings can be padded together as a single batch tensor (more efficient than
    per-sample tokenisation).
    """
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Exp27Trainer:
    """
    Encapsulates the full F2 training loop for Exp-27.

    Key design choices:
      • Gradient accumulation (accum_steps=4) simulates a larger effective
        batch without requiring proportionally more VRAM.
      • Mixed-precision (GradScaler + autocast) halves memory footprint and
        speeds up training on modern GPUs.
      • Per-epoch checkpointing supports resumable training via --resume_epoch.
    """

    def __init__(self, device: torch.device, max_neg: int = 15,
                 lr: float = 5e-5, max_length_q: int = 128,
                 max_length_p: int = 64, max_length_e: int = 64):
        self.device       = device
        self.max_neg      = max_neg        # max hard negatives per question
        self.max_length_q = max_length_q   # question token budget
        self.max_length_p = max_length_p   # path string token budget
        self.max_length_e = max_length_e   # entity name token budget
        self.tok   = AutoTokenizer.from_pretrained(ENCODER_NAME)
        self.model = PathAwareRanker().to(device)
        # AdamW with mild weight decay is standard for fine-tuning transformers
        self.opt   = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        print(f"[Exp27] Loaded PathAwareRanker ({ENCODER_NAME})  |  lr={lr}")

    def train(self, dataset: ListwiseS2Dataset, loader: DataLoader,
              epochs: int = 5, accum_steps: int = 4, start_epoch: int = 0):
        """
        Main training loop.

        For each batch the trainer:
          1. Samples gold + ≤max_neg negatives for every question.
          2. Tokenises question / path / entity strings together (shared padding).
          3. Runs a single forward pass for all candidates across all questions
             in the batch (amortises tokenisation overhead).
          4. Slices per-question score vectors using `offsets` and computes the
             listwise KL loss for each, then averages across questions.
          5. Scales and back-propagates the loss (AMP), stepping the optimiser
             every `accum_steps` micro-steps.

        Parameters
        ----------
        dataset     : ListwiseS2Dataset
        loader      : DataLoader  (batch_size=4, shuffle=True)
        epochs      : int   — total training epochs
        accum_steps : int   — gradient accumulation steps (effective batch×4)
        start_epoch : int   — epoch to resume from (0 = fresh start)
        """
        metrics_dir  = os.path.join(ROOT, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, METRICS_CSV)
        # Append mode when resuming; write mode (creates header) when starting fresh
        with open(metrics_path, "a" if start_epoch > 0 else "w") as f:
            if start_epoch == 0:
                f.write("epoch,avg_loss\n")

        # GradScaler prevents fp16 underflow when using autocast
        scaler = GradScaler("cuda")
        print(f"\n[Exp27 S2] Training  |  epochs={epochs}  "
              f"accum={accum_steps}  effective_batch={loader.batch_size * accum_steps}")

        for ep in range(start_epoch, epochs):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                # ── Build flat tokenisation lists across the mini-batch ────────
                # We flatten all candidates from all questions in the batch into
                # three parallel lists (q, p, e) and record per-question offsets
                # so we can later slice out the per-question score vectors.
                all_q, all_p, all_e, offsets = [], [], [], []
                offset = 0

                for item in batch:
                    q    = str(item["question"])
                    path = str(item.get("path") or "")
                    golds = [c for c in item["candidates"] if     c["is_gold"]]
                    negs  = [c for c in item["candidates"] if not c["is_gold"]]
                    if not golds or not negs:
                        continue   # skip degenerate samples

                    # Gold always at index 0 (required by loss_kl_distill assumption)
                    cands = golds[:1] + random.sample(negs, min(self.max_neg, len(negs)))
                    N = len(cands)

                    # Repeat question and path N times (one entry per candidate)
                    all_q.extend([q]    * N)
                    all_p.extend([path] * N)
                    all_e.extend([str(c.get("name", "")) for c in cands])
                    # Track the slice [offset, offset+N) for this question
                    offsets.append((offset, offset + N))
                    offset += N

                if not all_q:
                    continue   # entire mini-batch had no valid samples

                # ── Tokenise all inputs in one pass (batch padding) ────────────
                qe = self.tok(all_q, padding=True, truncation=True,
                              max_length=self.max_length_q,
                              return_tensors="pt").to(self.device)
                pe = self.tok(all_p, padding=True, truncation=True,
                              max_length=self.max_length_p,
                              return_tensors="pt").to(self.device)
                ee = self.tok(all_e, padding=True, truncation=True,
                              max_length=self.max_length_e,
                              return_tensors="pt").to(self.device)

                # ── Forward pass (mixed precision) ────────────────────────────
                with autocast("cuda"):
                    # scores shape: [total_candidates_in_batch]
                    scores = self.model(
                        qe["input_ids"], qe["attention_mask"],
                        pe["input_ids"], pe["attention_mask"],
                        ee["input_ids"], ee["attention_mask"],
                    )
                    # Slice per-question score vectors and compute KL loss for each;
                    # mean over questions in the batch, then divide by accum_steps
                    # so that accumulated gradients equal a full-batch average.
                    loss = torch.stack([
                        loss_kl_distill(scores[s:e]) for s, e in offsets
                    ]).mean() / accum_steps

                # ── Backward pass + optimiser step (with AMP scaling) ─────────
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(self.opt)
                    scaler.update()
                    self.opt.zero_grad()

                # Undo the /accum_steps division for logging purposes
                total_loss += loss.item() * accum_steps
                n_batches  += 1
                pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

            avg = total_loss / max(n_batches, 1)
            print(f"  Ep{ep+1} avg_loss: {avg:.4f}")
            with open(metrics_path, "a") as f:
                f.write(f"{ep+1},{avg:.4f}\n")

            # Per-epoch checkpoint for resumable training
            ep_ckpt = os.path.join(ROOT, "checkpoints", f"exp27_s2_epoch{ep+1}.pt")
            torch.save(self.model.state_dict(), ep_ckpt)
            print(f"  Checkpoint saved -> {ep_ckpt}")

        # Final checkpoint (overwrite for convenience)
        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"[Exp27] Final checkpoint -> {final_ckpt}")
        return final_ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Isolated Hit@1 evaluation on dev set
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: PathAwareRanker, tok, dataset: ListwiseS2Dataset,
             device: torch.device) -> float:
    """
    Measures isolated Stage 2 Hit@1: can Stage 2 alone rank the gold
    entity at position #1 from the set of candidates in the dev JSON?
    (Useful for ablation; E2E accuracy is what ultimately matters.)

    Implementation note: candidates are scored in chunks of 32 to avoid OOM
    when the dev set contains questions with very large candidate pools.

    Parameters
    ----------
    model   : trained PathAwareRanker
    tok     : the MPNet AutoTokenizer
    dataset : ListwiseS2Dataset wrapping the dev JSON
    device  : torch.device

    Returns
    -------
    float  — Hit@1 rate (fraction of questions where top-scored == gold)
    """
    model.eval()
    hits, total = 0, 0
    # batch_size=1 for evaluation: we must process one question at a time
    # since candidate counts vary and we need exact gold_idx tracking.
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc="[Exp27] Isolated Eval"):
        item  = batch[0]
        q     = str(item["question"])
        path  = str(item.get("path") or "")
        cands = item["candidates"]
        if not cands:
            continue

        # Find the index of the first gold candidate
        gold_idx = next(
            (i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None:
            continue   # no gold in dev record — skip

        names  = [str(c.get("name", "")) for c in cands]
        paths  = [path] * len(cands)   # same path for all candidates
        qs     = [q]    * len(cands)

        # Score candidates in chunks to handle large candidate sets without OOM
        CHUNK = 32
        all_scores = []
        for i in range(0, len(cands), CHUNK):
            qe = tok(qs[i:i+CHUNK],   padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to(device)
            pe = tok(paths[i:i+CHUNK], padding=True, truncation=True,
                     max_length=64,  return_tensors="pt").to(device)
            ee = tok(names[i:i+CHUNK], padding=True, truncation=True,
                     max_length=64,  return_tensors="pt").to(device)
            chunk_scores = model(qe["input_ids"], qe["attention_mask"],
                                 pe["input_ids"], pe["attention_mask"],
                                 ee["input_ids"], ee["attention_mask"])
            all_scores.append(chunk_scores)

        # Concatenate chunk scores and check if the argmax matches gold_idx
        scores = torch.cat(all_scores, dim=0)
        if torch.argmax(scores).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"[Exp27 Isolated Eval] Stage 2 Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


# ──────────────────────────────────────────────────────────────────────────────
# Main — CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Resume from a specific epoch checkpoint (avoids restarting from scratch)
    parser.add_argument("--resume_epoch", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, run isolated eval on dev set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp27] Device: {device}")

    train_file = os.path.join(ROOT, TRAIN_FILE)
    dev_file   = os.path.join(ROOT, DEV_FILE)

    if not os.path.exists(train_file) and not args.eval_only:
        raise FileNotFoundError(
            f"\n[Exp27] Training data not found: {train_file}\n"
            "  Run `python train/exp18_hard_negative_mining.py` first.\n")

    trainer = Exp27Trainer(device)

    if args.eval_only:
        # Load the final checkpoint and skip to evaluation
        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(f"[Exp27] Checkpoint not found: {final_ckpt}")
        trainer.model.load_state_dict(
            torch.load(final_ckpt, map_location=device))
    else:
        train_ds     = ListwiseS2Dataset(train_file, max_neg=15)
        # batch_size=4 × accum_steps=4 → effective batch of 16 questions
        train_loader = DataLoader(
            train_ds, batch_size=4, shuffle=True,
            collate_fn=collate_passthrough, pin_memory=True)

        start_epoch = 0
        if args.resume_epoch is not None:
            # Attempt to restore from a per-epoch checkpoint
            ep_ckpt = os.path.join(ROOT, "checkpoints",
                                   f"exp27_s2_epoch{args.resume_epoch}.pt")
            if os.path.exists(ep_ckpt):
                print(f"[Exp27] Resuming from epoch {args.resume_epoch}: {ep_ckpt}")
                trainer.model.load_state_dict(
                    torch.load(ep_ckpt, map_location=device))
                start_epoch = args.resume_epoch
            else:
                print(f"[Exp27] Epoch checkpoint not found; starting from scratch.")

        trainer.train(train_ds, train_loader,
                      epochs=5, accum_steps=4, start_epoch=start_epoch)

    # Run isolated dev-set evaluation regardless of training/eval_only mode
    if os.path.exists(dev_file):
        dev_ds = ListwiseS2Dataset(dev_file, max_neg=15)
        evaluate(trainer.model, trainer.tok, dev_ds, device)
    else:
        print(f"[Exp27] Dev file not found ({dev_file}) — skipping isolated eval.")


if __name__ == "__main__":
    main()
