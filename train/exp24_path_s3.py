"""
Exp 24: Stage 3 Training — Path-Aware Input (v7)
==================================================

Identical to Exp 23 in architecture (BGE reranker-base + KL-Distillation loss),
but simplifies the candidate string to:

    entity_name | path_nl

Previous Exp 17/23 used:  entity_name | path_nl | entity_type
Here we drop entity_type to isolate the pure contribution of the path and
avoid polluting the input with often-empty type strings.

This is registered as Stage 3 version 'v7' in the CDS pipeline.

Checkpoint: checkpoints/exp24_s3_path_v7.pt
Metrics:    metrics/exp24_s3_path_v7.csv
"""

import os, sys, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

from cds_pipeline.utils import path_to_nl

MODEL_NAME  = "BAAI/bge-reranker-base"
CKPT_NAME   = "exp24_s3_path_v7.pt"
METRICS_CSV = "exp24_s3_path_v7.csv"
TRAIN_FILE  = "data/exp18_cds_train_hard_full.json"
DEV_FILE    = "data/exp16_cds_dev.json"


# ── Input formatting ──────────────────────────────────────────────────────────

def build_candidate_str(entity_name: str, path_nl: str) -> str:
    """Format: 'Entity Name | film directed by → person spouse'"""
    parts = [entity_name.strip() if entity_name.strip() else "[UNK]"]
    if path_nl.strip():
        parts.append(path_nl.strip())
    return " | ".join(parts)


# ── KL-Distillation listwise loss ─────────────────────────────────────────────

def loss_kl_distill(scores: torch.Tensor) -> torch.Tensor:
    """Gold at index 0; soft-label teacher = 1.0 for gold, 0.1/(N-1) for negs."""
    N = scores.shape[0]
    teacher = torch.full((N,), 0.1 / max(N - 1, 1), device=scores.device)
    teacher[0] = 1.0
    teacher = teacher / teacher.sum()
    return F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction="sum")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PathS3Dataset(Dataset):
    def __init__(self, json_path: str, max_neg: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_neg = max_neg
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp24 Dataset] {len(self.samples)} samples with gold labels "
              f"from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def get_candidate_str(self, candidate: dict, item_path: str) -> str:
        # Use per-candidate path if available (beam search), else item-level path
        path_str = candidate.get("path") or item_path or ""
        path_nl  = path_to_nl(path_str)
        return build_candidate_str(candidate.get("name", ""), path_nl)


def collate_passthrough(batch):
    return batch


# ── Trainer ───────────────────────────────────────────────────────────────────

class Exp24Trainer:
    def __init__(self, device: torch.device, max_neg: int = 15,
                 lr: float = 5e-6, max_length: int = 192):
        self.device     = device
        self.max_neg    = max_neg
        self.max_length = max_length
        self.tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME).to(device)
        self.opt   = AdamW(self.model.parameters(), lr=lr)
        print(f"[Exp24] Loaded {MODEL_NAME}  |  max_length={max_length}  lr={lr}")

    def train(self, dataset: PathS3Dataset, loader: DataLoader,
              epochs: int = 5, accum_steps: int = 16, start_epoch: int = 0):

        metrics_dir  = os.path.join(ROOT, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, METRICS_CSV)
        with open(metrics_path, "a" if start_epoch > 0 else "w") as f:
            if start_epoch == 0:
                f.write("epoch,avg_loss\n")

        scaler = GradScaler("cuda")
        print(f"\n[Exp24 S3] Training  |  epochs={epochs}  "
              f"accum={accum_steps}  effective_batch={loader.batch_size * accum_steps}")

        for ep in range(start_epoch, epochs):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                all_qs, all_es, offsets = [], [], []
                offset = 0

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
                        all_qs.append(q)
                        all_es.append(dataset.get_candidate_str(c, item_path))

                    offsets.append((offset, offset + N))
                    offset += N

                if not all_qs:
                    continue

                enc = self.tok(
                    all_qs, all_es,
                    padding=True, truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                with autocast("cuda"):
                    logits = self.model(**enc).logits.squeeze(-1)
                    loss = torch.stack([
                        loss_kl_distill(logits[s:e]) for s, e in offsets
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

            ep_ckpt = os.path.join(ROOT, "checkpoints", f"exp24_s3_epoch{ep+1}.pt")
            torch.save(self.model.state_dict(), ep_ckpt)
            print(f"  Checkpoint saved -> {ep_ckpt}")

        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"[Exp24] Final checkpoint -> {final_ckpt}")
        return final_ckpt


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, dataset: PathS3Dataset,
             device: torch.device, max_length: int = 192) -> float:
    model.eval()
    hits, total = 0, 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc="[Exp24] Isolated Eval"):
        item      = batch[0]
        q         = str(item["question"])
        item_path = item.get("path") or ""
        cands     = item["candidates"]
        if not cands:
            continue

        gold_idx = next(
            (i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None:
            continue

        all_qs = [q] * len(cands)
        all_es = [dataset.get_candidate_str(c, item_path) for c in cands]

        logits_list = []
        for start in range(0, len(cands), 32):
            enc = tok(
                all_qs[start:start+32], all_es[start:start+32],
                padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            chunk = model(**enc).logits.squeeze(-1)
            if chunk.ndim == 0:
                chunk = chunk.unsqueeze(0)
            logits_list.append(chunk)

        logits = torch.cat(logits_list, dim=0)
        if torch.argmax(logits).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"[Exp24 Isolated Eval] Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Resume from epoch checkpoint, e.g. --resume_epoch 3")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; only run isolated evaluation on dev set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp24] Device: {device}")

    train_file = os.path.join(ROOT, TRAIN_FILE)
    dev_file   = os.path.join(ROOT, DEV_FILE)

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"\n[Exp24] Training data not found: {train_file}\n"
            "  Run `python train/exp18_hard_negative_mining.py` first.\n")

    trainer = Exp24Trainer(device)

    if args.eval_only:
        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(f"[Exp24] Checkpoint not found: {final_ckpt}")
        trainer.model.load_state_dict(
            torch.load(final_ckpt, map_location=device))
    else:
        train_ds     = PathS3Dataset(train_file, max_neg=15)
        train_loader = DataLoader(
            train_ds, batch_size=4, shuffle=True,
            collate_fn=collate_passthrough, pin_memory=True)

        start_epoch = 0
        if args.resume_epoch is not None:
            ep_ckpt = os.path.join(ROOT, "checkpoints",
                                   f"exp24_s3_epoch{args.resume_epoch}.pt")
            if os.path.exists(ep_ckpt):
                print(f"[Exp24] Resuming from epoch {args.resume_epoch}: {ep_ckpt}")
                trainer.model.load_state_dict(
                    torch.load(ep_ckpt, map_location=device))
                start_epoch = args.resume_epoch
            else:
                print(f"[Exp24] Epoch checkpoint not found, starting from scratch.")

        trainer.train(train_ds, train_loader,
                      epochs=5, accum_steps=16, start_epoch=start_epoch)

    if os.path.exists(dev_file):
        dev_ds = PathS3Dataset(dev_file, max_neg=15)
        evaluate(trainer.model, trainer.tok, dev_ds, device)
    else:
        print(f"[Exp24] Dev file not found ({dev_file}) — skipping isolated eval.")


if __name__ == "__main__":
    main()
