"""
Exp 34: T5 Listwise Semantic Scorer — Stage 3 Training
========================================================

Uses T5-base ENCODER (not the full seq2seq model) as a backbone.
Adds a Linear(768 → 1) scoring head on top of the CLS representation.

For each question + its ~16 candidates, the model outputs a scalar score
per candidate. Training loss = KL-Divergence between:
  - model's softmax score distribution
  - pre-computed soft semantic similarity labels (from MPNet embeddings)

This directly implements the user insight: teach a *ranking function* grounded
in semantic similarity geometry, not exact string generation.

Checkpoint: checkpoints/exp34_s3_listwise.pt
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, T5EncoderModel
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

ENCODER_NAME = "google/flan-t5-base"   # encoder only — no decoder loaded
TRAIN_FILE   = os.path.join(ROOT, "data", "exp34_s3_listwise_train.json")
CKPT_OUT     = os.path.join(ROOT, "checkpoints", "exp34_s3_listwise.pt")
MAX_NEG      = 15    # negatives per training sample
MAX_LEN_Q    = 128
MAX_LEN_E    = 64    # candidate name + path tokens


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class T5ListwiseScorer(nn.Module):
    """
    T5 encoder + linear scoring head.

    Input:  "[question] | [candidate_name] | [path]"  (one string per candidate)
    Output: scalar relevance score per candidate

    The encoder's mean-pooled representation is projected to a scalar.
    At inference: argmax over all candidate scores.
    """

    def __init__(self, encoder_name: str = ENCODER_NAME):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.d_model  # 768 for T5-base

        # Freeze bottom 6 of 12 encoder layers for memory efficiency
        for i, layer in enumerate(self.encoder.encoder.block):
            if i < 6:
                for p in layer.parameters():
                    p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [N, seq_len]   N = number of candidates in batch
            attention_mask: [N, seq_len]
        Returns:
            scores: [N] — one scalar per candidate
        """
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding tokens (cast to float32 to prevent fp16 overflow during sum)
        token_embs = enc_out.last_hidden_state.float()                  # [N, seq, 768]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [N, 768]
        return self.head(pooled).squeeze(-1)                             # [N]


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ListwiseS3Dataset(Dataset):
    def __init__(self, json_path: str, max_neg: int = MAX_NEG):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_neg = max_neg
        # Keep only samples that have both a gold and at least one negative
        self.samples = [
            s for s in raw
            if any(c.get("is_gold") for c in s["candidates"])
            and any(not c.get("is_gold") for c in s["candidates"])
        ]
        print(f"[Exp34 Dataset] {len(self.samples)} valid samples loaded from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_passthrough(batch):
    return batch


# ─────────────────────────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────────────────────────

def listwise_kl_loss(scores: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """
    KL-divergence between the model's score distribution and the soft semantic labels.

    Args:
        scores:      [N] raw logits from the model (float32)
        soft_labels: [N] pre-computed soft similarity distribution (sums to 1)
    """
    # Work in float32 to avoid fp16 underflow
    scores = scores.float()
    soft_labels = soft_labels.float()

    # Epsilon smoothing: prevent log(0) in KL-div by ensuring no zero entries
    eps = 1e-8
    soft_labels = soft_labels + eps
    soft_labels = soft_labels / soft_labels.sum()  # renormalize

    log_probs = F.log_softmax(scores, dim=0)
    loss = F.kl_div(log_probs, soft_labels, reduction="sum")

    # Guard against NaN (should not happen after smoothing, but safety net)
    if torch.isnan(loss):
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, device, epochs=3, lr=3e-5, accum_steps=16, batch_size=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_passthrough, num_workers=0, pin_memory=True)

    # Only optimize parameters with requires_grad=True (top 6 layers + head)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=0.01)
    scaler = GradScaler("cuda", init_scale=256)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Exp34] Total params: {total_params/1e6:.1f}M  |  Trainable: {trainable_params/1e6:.1f}M")
    print(f"[Exp34] Training for {epochs} epoch(s), lr={lr}, accum={accum_steps}")

    model.train()
    for ep in range(epochs):
        total_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")

        for step, batch in enumerate(pbar):
            # Build per-sample input strings and soft label tensors
            all_inputs, all_soft_labels, offsets = [], [], []
            offset = 0

            for item in batch:
                q    = str(item.get("question", ""))
                path = str(item.get("path", "") or "")
                cands = item["candidates"]

                golds = [c for c in cands if c.get("is_gold")]
                negs  = [c for c in cands if not c.get("is_gold")]
                if not golds or not negs:
                    continue

                # Sample a fixed set: 1 gold + up to MAX_NEG negatives
                selected = golds[:1] + random.sample(negs, min(MAX_NEG, len(negs)))
                random.shuffle(selected)  # shuffle so gold isn't always at idx 0

                for c in selected:
                    name      = str(c.get("name", "") or "").strip()
                    cand_path = str(c.get("path", "") or path).strip()
                    # Input: "question | name | path" — T5 sees full context per candidate
                    all_inputs.append(f"{q} | {name} | {cand_path}" if cand_path else f"{q} | {name}")

                # Soft labels: pre-computed semantic similarity from dataset builder
                soft_label_vals = torch.tensor(
                    [c.get("soft_label", 0.01) for c in selected], dtype=torch.float32
                )
                # Re-normalize in case of floating point drift
                soft_label_vals = (soft_label_vals / soft_label_vals.sum()).to(device)
                all_soft_labels.append(soft_label_vals)

                offsets.append((offset, offset + len(selected)))
                offset += len(selected)

            if not all_inputs:
                continue

            # Tokenize all candidates across the batch in one call
            enc = tokenizer(
                all_inputs,
                padding=True,
                truncation=True,
                max_length=MAX_LEN_Q,
                return_tensors="pt"
            ).to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                scores = model(enc["input_ids"], enc["attention_mask"])  # [total_N]

            # Compute listwise KL loss per sample, average across batch
            loss = torch.stack([
                listwise_kl_loss(scores[s:e].float(), all_soft_labels[i])
                for i, (s, e) in enumerate(offsets)
            ]).mean() / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            n_batches  += 1
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch {ep+1} avg loss: {avg:.4f}")

        # Save per-epoch checkpoint
        ep_ckpt = CKPT_OUT.replace(".pt", f"_ep{ep+1}.pt")
        torch.save(model.state_dict(), ep_ckpt)
        print(f"  Saved: {ep_ckpt}")

    # Save final
    torch.save(model.state_dict(), CKPT_OUT)
    print(f"\n[Exp34] Final checkpoint saved: {CKPT_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=3)
    parser.add_argument("--lr",      type=float, default=1e-5)  # lowered from 3e-5 to prevent NaN
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp34] Device: {device}")

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_FILE}\n"
            "Run train/exp34_build_s3_listwise_dataset.py first."
        )

    print(f"[Exp34] Loading T5 encoder: {ENCODER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model     = T5ListwiseScorer(ENCODER_NAME).to(device)

    # Warm-resume from latest checkpoint
    if not args.no_resume and os.path.exists(CKPT_OUT):
        model.load_state_dict(torch.load(CKPT_OUT, map_location=device), strict=False)
        print(f"[Exp34] Resumed weights from {CKPT_OUT}")

    dataset = ListwiseS3Dataset(TRAIN_FILE)
    train(model, tokenizer, dataset, device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
