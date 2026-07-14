"""
Exp 19: Enriched Stage 3 Input for BGE Cross-Encoder
=====================================================

HYPOTHESIS
----------
Current Stage 3 (exp16v2) gives the cross-encoder only:
    question | entity_name
e.g. "Who directed the Saving Private Ryan? | Steven Spielberg"

BGE cannot distinguish two candidates with similar names reached via
different graph paths, because it sees no graph context at all.

Fix: construct a richer candidate string on the fly from data that is
already available — the traversed path and the entity type from the
CDS dataset JSON:
    entity_name | path_as_natural_language | entity_type
e.g. "Steven Spielberg | film directed by → director works | person"

Path NL conversion is derived from the relation2id vocab that is already
on disk (data/processed_entity/relation2id.pt) using the same heuristic
as RelationEmbeddingBank._rel_to_text in exp15_strl.py.
No hand-crafted mapping. No hardcoding.

WHAT CHANGES FROM exp16v2
--------------------------
* Only Stage 3 input construction changes.
* Loss function, model architecture, LR, accum steps are identical.
* max_length raised from 128 → 192 to fit the richer string.

EXPECTED GAIN: +3–5% Hit@1
CHECKPOINT:    checkpoints/exp19_s3_enriched.pt
METRICS:       metrics/exp19_s3_enriched.csv
"""

import os, sys, json, random
import torch
import torch.nn as nn
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
    """
    Build the enriched document string for the cross-encoder.
    Fields are pipe-separated so the encoder can use token boundaries.
    Empty fields are omitted to keep the string short.
    """
    parts = [entity_name.strip()] if entity_name.strip() else ["[UNK]"]
    if path_nl.strip():
        parts.append(path_nl.strip())
    if entity_type.strip():
        parts.append(entity_type.strip())
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
#  Loss  (same as exp16v2 Stage 3 winner)
# ─────────────────────────────────────────────────────────────

def loss_kl_distill(scores: torch.Tensor) -> torch.Tensor:
    """
    KL-divergence distillation.
    Gold (index 0) gets soft label 1.0.
    Negatives share 0.1 uniformly.
    Proved best (57.5% Hit@1) in the exp16 ablation.
    """
    N = scores.shape[0]
    teacher = torch.full((N,), 0.1 / max(N - 1, 1), device=scores.device)
    teacher[0] = 1.0
    teacher = teacher / teacher.sum()
    return F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction="sum")


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class EnrichedCDSDataset(Dataset):
    """
    Reads the same JSON format as exp16v2 (exp18_cds_train_hard.json).

    Expected schema per item:
    {
        "question":   "...",
        "path":       "film.film.directed_by people.person.nationality",  # optional
        "candidates": [
            {
                "name":      "Steven Spielberg",
                "is_gold":   true,
                "path":      "film.film.directed_by",   # candidate-level path (optional)
                "type":      "person"                   # entity type string (optional)
            },
            ...
        ]
    }
    Candidate-level "path" takes priority over item-level "path" if both exist.
    """
    def __init__(self, json_path: str, id2rel: dict, max_neg: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.id2rel  = id2rel
        self.max_neg = max_neg
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp17 Dataset] {len(self.samples)} samples with gold labels "
              f"from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def get_enriched_str(self, candidate: dict, item_path: str) -> str:
        path_str  = candidate.get("path") or item_path or ""
        path_nl   = path_to_nl(path_str)
        ent_type  = candidate.get("type") or candidate.get("entity_type") or ""
        return build_enriched_candidate_str(
            candidate.get("name", ""), path_nl, ent_type)


def collate_passthrough(batch):
    return batch


# ─────────────────────────────────────────────────────────────
#  Stage 3 Trainer
# ─────────────────────────────────────────────────────────────

class EnrichedStage3Trainer:
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
        print(f"[Exp17] Loaded {model_name}  |  max_length={max_length}")

    def train(self, dataset: EnrichedCDSDataset, loader: DataLoader,
              epochs: int = 5, accum_steps: int = 16):

        metrics_dir = os.path.join(ROOT, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "exp19_s3_hard_negatives.csv")
        with open(metrics_path, "w") as f:
            f.write("epoch,avg_loss\n")

        scaler = GradScaler("cuda")
        print(f"\n[Exp17 S3] Training  |  epochs={epochs}  "
              f"accum={accum_steps}  effective_batch={loader.batch_size * accum_steps}")

        for ep in range(epochs):
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

        ckpt = os.path.join(ROOT, "checkpoints", "exp19_s3_hard_negatives.pt")
        torch.save(self.model.state_dict(), ckpt)
        print(f"[Exp19] Checkpoint saved -> {ckpt}")


# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, dataset: EnrichedCDSDataset,
             device: torch.device, max_length: int = 192) -> float:
    """
    Hit@1: score every candidate for each question,
    check whether the gold candidate is ranked first.
    """
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

        enc = tok(all_qs, all_es, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits.squeeze(-1)
        if torch.argmax(logits).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"[Exp17 Eval] Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp17] Device: {device}")

    # ── Load vocab (relation2id already on disk from exp7/exp9) ──────────────
    rel2id_path = os.path.join(ROOT, "data/processed_entity/relation2id.pt")
    if not os.path.exists(rel2id_path):
        raise FileNotFoundError(
            f"relation2id.pt not found at {rel2id_path}. "
            "Run exp7_roberta.py data preparation first.")
    rel2id = torch.load(rel2id_path)
    id2rel = {v: k for k, v in rel2id.items()}   # int → str, used for path NL

    # ── Paths ─────────────────────────────────────────────────────────────────
    train_file = os.path.join(ROOT, "data/exp18_cds_train_hard.json")
    dev_file   = os.path.join(ROOT, "data/exp16_cds_dev.json")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "Use the same CDS dataset as exp16v2.")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    train_ds = EnrichedCDSDataset(train_file, id2rel, max_neg=15)
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=collate_passthrough, pin_memory=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = EnrichedStage3Trainer(device)
    trainer.train(train_ds, train_loader, epochs=5, accum_steps=16)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if os.path.exists(dev_file):
        dev_ds = EnrichedCDSDataset(dev_file, id2rel, max_neg=15)
        evaluate(trainer.model, trainer.tok, dev_ds, device)
    else:
        print(f"[Exp17] Dev file not found at {dev_file} — skipping eval.")
        print("         Run evaluation manually using the saved checkpoint.")


if __name__ == "__main__":
    main()
