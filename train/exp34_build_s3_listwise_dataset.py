"""
Exp 34: Build Soft-Label Listwise Dataset for Stage 3
=======================================================

Takes the existing hard-negative training data (exp18_cds_train_hard_full.json)
and adds soft semantic similarity labels to each candidate.

The soft label for a candidate = cosine_sim(candidate_name_emb, gold_name_emb),
computed using a frozen MPNet sentence-transformer (same model used in Stage 2).

This converts binary (gold=1, neg=0) hard labels into a smooth distribution
that captures semantic proximity — a director who is wrong is still "closer"
to the gold director than a film title is.

Output: data/exp34_s3_listwise_train.json
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

INPUT_FILE  = os.path.join(ROOT, "data", "exp18_cds_train_hard_full.json")
OUTPUT_FILE = os.path.join(ROOT, "data", "exp34_s3_listwise_train.json")
ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"

BATCH_SIZE = 256   # entity names to embed at once
TEMPERATURE = 0.5  # controls how peaked the soft label distribution is


def mean_pool(model_output, attention_mask):
    """Mean pooling over token embeddings (standard sentence-transformer pooling)."""
    token_embs = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return torch.sum(token_embs * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


@torch.no_grad()
def embed_strings(strings, tokenizer, model, device):
    """Embed a list of strings into normalized 768-dim vectors using MPNet."""
    all_embs = []
    for i in range(0, len(strings), BATCH_SIZE):
        batch = strings[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)
        out = model(**enc)
        embs = mean_pool(out, enc["attention_mask"])
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp34 Dataset] Device: {device}")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run exp18_hard_negative_mining.py first."
        )

    print(f"[Exp34 Dataset] Loading MPNet encoder: {ENCODER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AutoModel.from_pretrained(ENCODER_NAME).to(device)
    model.eval()

    print(f"[Exp34 Dataset] Loading training data from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"[Exp34 Dataset] Processing {len(raw_data)} samples...")

    output_samples = []
    skipped = 0

    for item in tqdm(raw_data, desc="Computing soft labels"):
        question = str(item.get("question", ""))
        path     = str(item.get("path", "") or "")
        candidates = item.get("candidates", [])

        # Need at least one gold and one negative
        golds = [c for c in candidates if c.get("is_gold")]
        negs  = [c for c in candidates if not c.get("is_gold")]
        if not golds or not negs:
            skipped += 1
            continue

        gold_name = str(golds[0].get("name", "") or "").strip()
        if not gold_name:
            skipped += 1
            continue

        cand_names = [str(c.get("name", "") or "").strip() for c in candidates]

        # Embed gold name and all candidates
        all_names = [gold_name] + cand_names
        embs = embed_strings(all_names, tokenizer, model, device)

        gold_emb  = embs[0:1]              # [1, 768]
        cand_embs = embs[1:]               # [N, 768]

        # Cosine similarity: each candidate vs gold
        sims = F.cosine_similarity(cand_embs, gold_emb.expand_as(cand_embs))  # [N]

        # Clip negatives to [0, 1] — no negative similarity scores as labels
        sims = sims.clamp(min=0.0)

        # Apply temperature and store soft labels
        soft_labels = (sims / TEMPERATURE).softmax(dim=0).tolist()

        # Build output candidates with soft labels attached
        output_cands = []
        for c, soft_label in zip(candidates, soft_labels):
            output_cands.append({
                "mid":       c.get("mid", ""),
                "name":      c.get("name", ""),
                "path":      c.get("path", path),
                "is_gold":   c.get("is_gold", False),
                "soft_label": soft_label,   # ← new field
            })

        output_samples.append({
            "question":   question,
            "path":       path,
            "candidates": output_cands,
        })

    print(f"\n[Exp34 Dataset] Done. {len(output_samples)} samples written, {skipped} skipped.")
    print(f"[Exp34 Dataset] Saving to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False)

    print("[Exp34 Dataset] Complete.")


if __name__ == "__main__":
    main()
