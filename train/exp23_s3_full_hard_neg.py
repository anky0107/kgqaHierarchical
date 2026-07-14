"""
Exp 23: Stage 3 Training on Full 27k Hard Negatives
======================================================

Same architecture as Exp 19, but trained on the full hard negative dataset
generated from the 27k training questions (exp18_cds_train_hard_full.json).

Expected checkpoint: checkpoints/exp23_s3_full_hard_neg.pt
Expected metrics:    metrics/exp23_s3_full_hard_neg.csv
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


def build_enriched_candidate_str(entity_name: str, path_nl: str,
                                  entity_type: str) -> str:
    parts = [entity_name.strip()] if entity_name.strip() else ["[UNK]"]
    if path_nl.strip():
        parts.append(path_nl.strip())
    if entity_type.strip():
        parts.append(entity_type.strip())
    return " | ".join(parts)


def loss_kl_distill(scores: torch.Tensor) -> torch.Tensor:
    """KL-divergence distillation — gold index 0 gets soft label 1.0."""
    N = scores.shape[0]
    teacher = torch.full((N,), 0.1 / max(N - 1, 1), device=scores.device)
    teacher[0] = 1.0
    teacher = teacher / teacher.sum()
    return F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction="sum")


class EnrichedCDSDataset(Dataset):
    def __init__(self, json_path: str, max_neg: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_neg = max_neg
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp23 Dataset] {len(self.samples)} samples with gold labels "
              f"from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def get_enriched_str(self, candidate: dict, item_path: str) -> str:
        path_str = candidate.get("path") or item_path or ""
        path_nl  = path_to_nl(path_str)
        ent_type = candidate.get("type") or candidate.get("entity_type") or ""
        return build_enriched_candidate_str(
            candidate.get("name", ""), path_nl, ent_type)


def collate_passthrough(batch):
    return batch


class Exp23Trainer:
    def __init__(self, device: torch.device, max_neg: int = 15,
                 model_name: str = "BAAI/bge-reranker-base",
                 lr: float = 5e-6, max_length: int = 192):
        self.device     = device
        self.max_neg    = max_neg
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(device)
        self.opt = AdamW(self.model.parameters(), lr=lr)
        print(f"[Exp23] Loaded {model_name}  |  max_length={max_length}")

    def train(self, dataset: EnrichedCDSDataset, loader: DataLoader,
              epochs: int = 5, accum_steps: int = 16, start_epoch: int = 0):

        metrics_dir = os.path.join(ROOT, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "exp23_s3_full_hard_neg.csv")
        with open(metrics_path, "a" if start_epoch > 0 else "w") as f:
            if start_epoch == 0:
                f.write("epoch,avg_loss\n")

        scaler = GradScaler("cuda")
        print(f"\n[Exp23 S3] Training  |  epochs={epochs}  "
              f"accum={accum_steps}  effective_batch={loader.batch_size * accum_steps}")

        for ep in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0.0
            n_batches  = 0
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
                        enriched = dataset.get_enriched_str(c, item_path)
                        all_qs.append(q)
                        all_es.append(enriched)

                    offsets.append((offset, offset + N))
                    offset += N

                if not all_qs:
                    continue

                enc = self.tok(
                    all_qs, all_es,
                    padding=True, truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
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

            # Save epoch checkpoint so we don't lose progress
            ep_ckpt = os.path.join(ROOT, "checkpoints", f"exp23_s3_epoch{ep+1}.pt")
            torch.save(self.model.state_dict(), ep_ckpt)
            print(f"  Checkpoint saved -> {ep_ckpt}")

        ckpt = os.path.join(ROOT, "checkpoints", "exp23_s3_full_hard_neg.pt")
        torch.save(self.model.state_dict(), ckpt)
        print(f"[Exp23] Final checkpoint saved -> {ckpt}")
        return ckpt


@torch.no_grad()
def evaluate(model, tok, dataset: EnrichedCDSDataset,
             device: torch.device, max_length: int = 192) -> float:
    model.eval()
    hits = 0; total = 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc="Evaluating"):
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
        all_es = [dataset.get_enriched_str(c, item_path) for c in cands]

        # Chunk evaluation batches to prevent CUDA OOM
        logits_list = []
        chunk_size = 32
        for start_idx in range(0, len(cands), chunk_size):
            chunk_qs = all_qs[start_idx:start_idx + chunk_size]
            chunk_es = all_es[start_idx:start_idx + chunk_size]
            enc = tok(chunk_qs, chunk_es, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            logits_list.append(logits)
            
        logits = torch.cat(logits_list, dim=0)
        if torch.argmax(logits).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"[Exp23 Eval] Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Resume training from a specific epoch checkpoint (e.g. 3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp23] Device: {device}")

    train_file = os.path.join(ROOT, "data/exp18_cds_train_hard_full.json")
    dev_file   = os.path.join(ROOT, "data/exp16_cds_dev.json")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"\n[Exp23] Training data not found: {train_file}\n"
            "  Run `python train/exp18_hard_negative_mining.py` first!\n"
            "  (This generates hard negatives from the full 27k dataset.)")

    train_ds = EnrichedCDSDataset(train_file, max_neg=15)
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=collate_passthrough, pin_memory=True)

    trainer = Exp23Trainer(device)

    start_epoch = 0
    if args.resume_epoch is not None:
        ckpt_path = os.path.join(ROOT, "checkpoints", f"exp23_s3_epoch{args.resume_epoch}.pt")
        if os.path.exists(ckpt_path):
            print(f"[Exp23] Resuming from epoch {args.resume_epoch} checkpoint: {ckpt_path}")
            trainer.model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_epoch = args.resume_epoch
        else:
            print(f"[Exp23] Checkpoint not found: {ckpt_path}. Starting from scratch.")

    # Modify Trainer's loop to take start_epoch
    trainer.train(train_ds, train_loader, epochs=5, accum_steps=16, start_epoch=start_epoch)

    if os.path.exists(dev_file):
        dev_ds = EnrichedCDSDataset(dev_file, max_neg=15)
        evaluate(trainer.model, trainer.tok, dev_ds, device)
    else:
        print(f"[Exp23] Dev file not found at {dev_file} — skipping isolated eval.")


if __name__ == "__main__":
    main()

