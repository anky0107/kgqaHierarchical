"""
Exp 25: Stage 2 Listwise Training
===================================

Replaces the Soft-Margin (binary) loss in Stage 2 (PathAwareRanker) with
KL-Distillation Listwise Loss â€” the same loss that won the Stage 3 ablation.

The core insight: Stage 2 currently classifies each candidate independently
(BCE/SoftMargin). At inference it must RANK 200 candidates to pick the top 15.
These are different tasks. Listwise KL-Distill trains Stage 2 as an explicit
ranker over the full candidate set, preventing gold-entity loss in the 200â†’15 cut.

Architecture: Same PathAwareRanker (MPNet-base-v2 + MLP fusion) as before.
Dataset:      exp18_cds_train_hard_full.json (27k hard negatives â€” 1 gold + â‰¤15 negs)
Loss:         KL-Distillation (gold soft-label = 1.0, neg soft-label = 0.1/(N-1))

Checkpoint: checkpoints/exp27_s2_listwise.pt
Metrics:    metrics/exp27_s2_listwise.csv
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

ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
CKPT_NAME    = "exp27_s2_mpnet.pt"
METRICS_CSV  = "exp27_s2_mpnet.csv"
TRAIN_FILE   = "data/exp26_s2_hard_negatives.json"
DEV_FILE     = "data/exp16_cds_dev.json"


# â”€â”€ PathAwareRanker (canonical copy from cds_pipeline/models.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PathAwareRanker(nn.Module):
    """
    Stage 2: MPNet-base-v2 shared encoder + 3-input MLP fusion head.
    Score = MLP([q_emb; p_emb; e_emb])  â†’ scalar
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
        hidden = self.encoder.config.hidden_size          # 768
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
        enc = self.encoder
        q = enc(q_ids,  attention_mask=q_mask).last_hidden_state[:, 0, :]
        p = enc(p_ids,  attention_mask=p_mask).last_hidden_state[:, 0, :]
        e = enc(e_ids,  attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q, p, e], dim=-1)).squeeze(-1)


# â”€â”€ KL-Distillation listwise loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def loss_kl_distill(scores: torch.Tensor) -> torch.Tensor:
    """Gold at index 0; soft teacher: 1.0 for gold, 0.1/(N-1) for negatives."""
    N = scores.shape[0]
    teacher = torch.full((N,), 0.1 / max(N - 1, 1), device=scores.device)
    teacher[0] = 1.0
    teacher = teacher / teacher.sum()
    return F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction="sum")


# â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ListwiseS2Dataset(Dataset):
    def __init__(self, json_path: str, max_neg: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_neg = max_neg
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp27 Dataset] {len(self.samples)} samples with gold labels "
              f"from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_passthrough(batch):
    return batch


# â”€â”€ Trainer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class Exp27Trainer:
    def __init__(self, device: torch.device, max_neg: int = 15,
                 lr: float = 5e-5, max_length_q: int = 128,
                 max_length_p: int = 64, max_length_e: int = 64):
        self.device       = device
        self.max_neg      = max_neg
        self.max_length_q = max_length_q
        self.max_length_p = max_length_p
        self.max_length_e = max_length_e
        self.tok   = AutoTokenizer.from_pretrained(ENCODER_NAME)
        self.model = PathAwareRanker().to(device)
        self.opt   = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        print(f"[Exp27] Loaded PathAwareRanker ({ENCODER_NAME})  |  lr={lr}")

    def train(self, dataset: ListwiseS2Dataset, loader: DataLoader,
              epochs: int = 5, accum_steps: int = 4, start_epoch: int = 0):

        metrics_dir  = os.path.join(ROOT, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, METRICS_CSV)
        with open(metrics_path, "a" if start_epoch > 0 else "w") as f:
            if start_epoch == 0:
                f.write("epoch,avg_loss\n")

        scaler = GradScaler("cuda")
        print(f"\n[Exp27 S2] Training  |  epochs={epochs}  "
              f"accum={accum_steps}  effective_batch={loader.batch_size * accum_steps}")

        for ep in range(start_epoch, epochs):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                all_q, all_p, all_e, offsets = [], [], [], []
                offset = 0

                for item in batch:
                    q    = str(item["question"])
                    path = str(item.get("path") or "")
                    golds = [c for c in item["candidates"] if     c["is_gold"]]
                    negs  = [c for c in item["candidates"] if not c["is_gold"]]
                    if not golds or not negs:
                        continue

                    cands = golds[:1] + random.sample(negs, min(self.max_neg, len(negs)))
                    N = len(cands)

                    all_q.extend([q]    * N)
                    all_p.extend([path] * N)
                    all_e.extend([str(c.get("name", "")) for c in cands])
                    offsets.append((offset, offset + N))
                    offset += N

                if not all_q:
                    continue

                qe = self.tok(all_q, padding=True, truncation=True,
                              max_length=self.max_length_q,
                              return_tensors="pt").to(self.device)
                pe = self.tok(all_p, padding=True, truncation=True,
                              max_length=self.max_length_p,
                              return_tensors="pt").to(self.device)
                ee = self.tok(all_e, padding=True, truncation=True,
                              max_length=self.max_length_e,
                              return_tensors="pt").to(self.device)

                with autocast("cuda"):
                    scores = self.model(
                        qe["input_ids"], qe["attention_mask"],
                        pe["input_ids"], pe["attention_mask"],
                        ee["input_ids"], ee["attention_mask"],
                    )
                    loss = torch.stack([
                        loss_kl_distill(scores[s:e]) for s, e in offsets
                    ]).mean() / accum_steps

                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(self.opt)
                    scaler.update()
                    self.opt.zero_grad()

                total_loss += loss.item() * accum_steps
                n_batches  += 1
                pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

            avg = total_loss / max(n_batches, 1)
            print(f"  Ep{ep+1} avg_loss: {avg:.4f}")
            with open(metrics_path, "a") as f:
                f.write(f"{ep+1},{avg:.4f}\n")

            ep_ckpt = os.path.join(ROOT, "checkpoints", f"exp27_s2_epoch{ep+1}.pt")
            torch.save(self.model.state_dict(), ep_ckpt)
            print(f"  Checkpoint saved -> {ep_ckpt}")

        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"[Exp27] Final checkpoint -> {final_ckpt}")
        return final_ckpt


# â”€â”€ Isolated hit@1 evaluation on dev set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@torch.no_grad()
def evaluate(model: PathAwareRanker, tok, dataset: ListwiseS2Dataset,
             device: torch.device) -> float:
    """
    Measures isolated Stage 2 Hit@1: can Stage 2 alone rank the gold
    entity at position #1 from the set of candidates in the dev JSON?
    (Useful for ablation; E2E accuracy is what ultimately matters.)
    """
    model.eval()
    hits, total = 0, 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc="[Exp27] Isolated Eval"):
        item  = batch[0]
        q     = str(item["question"])
        path  = str(item.get("path") or "")
        cands = item["candidates"]
        if not cands:
            continue

        gold_idx = next(
            (i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None:
            continue

        names  = [str(c.get("name", "")) for c in cands]
        paths  = [path] * len(cands)
        qs     = [q]    * len(cands)

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

        scores = torch.cat(all_scores, dim=0)
        if torch.argmax(scores).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"[Exp27 Isolated Eval] Stage 2 Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    import argparse
    parser = argparse.ArgumentParser()
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
        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(f"[Exp27] Checkpoint not found: {final_ckpt}")
        trainer.model.load_state_dict(
            torch.load(final_ckpt, map_location=device))
    else:
        train_ds     = ListwiseS2Dataset(train_file, max_neg=15)
        train_loader = DataLoader(
            train_ds, batch_size=4, shuffle=True,
            collate_fn=collate_passthrough, pin_memory=True)

        start_epoch = 0
        if args.resume_epoch is not None:
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

    if os.path.exists(dev_file):
        dev_ds = ListwiseS2Dataset(dev_file, max_neg=15)
        evaluate(trainer.model, trainer.tok, dev_ds, device)
    else:
        print(f"[Exp27] Dev file not found ({dev_file}) â€” skipping isolated eval.")


if __name__ == "__main__":
    main()
