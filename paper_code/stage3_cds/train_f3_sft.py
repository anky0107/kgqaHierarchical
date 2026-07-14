"""
train_f3_sft.py — CDS Pipeline: F3 Generative Judge, Supervised Fine-Tuning (Exp-26)
======================================================================================

Paper Section: §V-E  "Cascading Dual-Stage Filtering (CDS) — F3 Listwise Generative Judge (SFT)"

Purpose
-------
Trains the third and final stage (F3) of the CDS pipeline using Supervised
Fine-Tuning (SFT).  F3 receives the top-50 candidates retained by the MPNet
F2 ranker (in practice the prompt is constructed with the top-15 for prompt
length reasons) and is trained as a **listwise generative judge** — it reads a
multi-candidate prompt and is asked to generate the exact name of the correct
answer entity as free text.

This SFT checkpoint (exp26_t5_generative_s3.pt) is the initialisation point
for the subsequent DPO alignment step (train_f3_dpo.py / Exp-38).

Architecture
------------
  Base model : google/flan-t5-base  (encoder-decoder, ~250M parameters)
  Training   : Sequence-to-sequence cross-entropy (teacher-forcing)
  Input      : Listwise MC prompt  (question + ranked candidate list with paths)
  Output     : Exact name of the gold entity (free-text generation)

The model is a generative encoder-decoder rather than a discriminative
cross-encoder, which means:
  • It can leverage instruction-following capabilities of the pre-trained Flan-T5.
  • The output is unconstrained text — at inference, the generated string is
    matched against the candidate list to select the final answer.

Pipeline position
-----------------
  [F2] top-50 candidates (15 used in prompt)
        │
        ▼
  [train_f3_sft.py]  ← THIS FILE  (SFT warm-up)
        │  checkpoint: checkpoints/exp26_t5_generative_s3.pt
        ▼
  [build_dpo_dataset.py]  construct preference pairs
        │
        ▼
  [train_f3_dpo.py]  DPO alignment (beta=0.1)

Inputs
------
- data/exp18_cds_train_hard_full.json  : hard-negative training set (27k records)
- data/exp16_cds_dev.json              : dev set (used optionally for evaluation)
  Each record: {"question": str, "path": ..., "candidates": [{name, is_gold, ...}]}

Outputs
-------
- checkpoints/exp26_t5_generative_s3.pt  : fine-tuned Flan-T5 weights (SFT)

Key hyperparameters
-------------------
- lr           : 1e-4  (higher than DPO because SFT is a more stable objective)
- batch_size   : 2     (small to avoid OOM with T5-base + long prompts)
- accum_steps  : 8     (effective batch = 2 × 8 = 16)
- epochs       : 3
- max_src_len  : 512 tokens  (prompt length budget)
- max_tgt_len  : 64 tokens   (target entity name)
- max_cands    : 15 candidates per prompt
- dtype        : bfloat16 (via autocast) — numerically stable for T5

Prompt format (build_prompt)
-----------------------------
  Question: <question>

  Candidates:
  1. <name1> (Path: <path_nl>)
  2. <name2>
  ...

  Which of the above candidates is the correct answer to the question?
  Answer with the exact name.

The path information is included when available so the model can reason about
the structural proximity of each candidate to the topic entity.

Training objective
------------------
Standard Seq2Seq cross-entropy (teacher forcing):
  loss = -Σ log P(gold_name_token | prefix, encoder_output)
Padding tokens in labels are replaced with -100 so they are masked from loss.
"""

import os, sys, json, random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

from cds_pipeline.utils import path_to_nl

# ── Global constants ──────────────────────────────────────────────────────────
MODEL_NAME  = "google/flan-t5-base"                       # F3 backbone (§V-E)
CKPT_NAME   = "exp26_t5_generative_s3.pt"                 # output checkpoint name
TRAIN_FILE  = "data/exp18_cds_train_hard_full.json"       # hard-negative training set
DEV_FILE    = "data/exp16_cds_dev.json"                   # dev set path

# ──────────────────────────────────────────────────────────────────────────────
# Input formatting — listwise MC prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, candidates: list, item_path: str) -> str:
    """
    Construct a natural-language multiple-choice prompt for the Flan-T5 judge.

    Format:
        Question: <question>

        Candidates:
        1. <name1> (Path: <path_nl>)   ← path included when available
        2. <name2>
        ...

        Which of the above candidates is the correct answer to the question?
        Answer with the exact name.

    Including the KG reasoning path in the prompt gives the model structural
    evidence about how each candidate was reached from the topic entity, helping
    it distinguish between semantically similar entities that differ only in their
    relation to the question topic.

    Parameters
    ----------
    question   : str   — natural-language question
    candidates : list  — list of candidate dicts (each has at least 'name' key)
    item_path  : str   — fallback path string if individual candidates lack one

    Returns
    -------
    str  — formatted prompt ready for tokenisation
    """
    prompt = f"Question: {question}\n\nCandidates:\n"
    for i, c in enumerate(candidates, 1):
        name     = c.get("name", "").strip() or "[UNK]"
        # Prefer per-candidate path; fall back to question-level path from F1
        path_str = c.get("path") or item_path or ""
        path_nl  = path_to_nl(path_str)   # convert relation IDs to natural language
        if path_nl:
            prompt += f"{i}. {name} (Path: {path_nl})\n"
        else:
            prompt += f"{i}. {name}\n"
    prompt += "\nWhich of the above candidates is the correct answer to the question? Answer with the exact name."
    return prompt

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GenerativeS3Dataset(Dataset):
    """
    PyTorch Dataset for the Flan-T5 SFT training.

    Loads the hard-negative candidate JSON and filters to samples that contain
    at least one gold-labelled candidate (necessary for constructing targets).

    Parameters
    ----------
    json_path  : str  — path to the candidate JSON file
    max_cands  : int  — maximum candidates to include in each prompt (default 15)
    """
    def __init__(self, json_path: str, max_cands: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_cands = max_cands
        # Retain only samples with at least one gold entity for supervision
        self.samples = [s for s in raw if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp26 Dataset] {len(self.samples)} samples from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

def collate_passthrough(batch):
    """
    Identity collate — returns the raw list of dicts.
    Tokenisation and prompt construction happen inside the training loop.
    """
    return batch

# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Exp26Trainer:
    """
    Encapsulates the Flan-T5 SFT training loop for Exp-26.

    Design decisions:
      • batch_size=2 with accum_steps=8 gives effective batch 16 while keeping
        GPU memory manageable for Flan-T5-base with 512-token inputs.
      • bfloat16 (via autocast) is preferred over float16 for T5 because bfloat16
        has the same exponent range as float32, avoiding the gradient underflow
        issues that can occur with fp16 on generative tasks.
      • Candidates are randomly shuffled at each training step (after ensuring
        gold is in the pool) to prevent the model from memorising positional biases
        (e.g., always picking "candidate 1").
    """

    def __init__(self, device: torch.device, lr: float = 1e-4, max_src_len: int = 512, max_tgt_len: int = 64):
        self.device      = device
        self.max_src_len = max_src_len   # max tokens for the MC prompt (source)
        self.max_tgt_len = max_tgt_len   # max tokens for the gold entity name (target)
        self.tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        # AdamW — standard for fine-tuning encoder-decoder transformers
        self.opt   = AdamW(self.model.parameters(), lr=lr)
        print(f"[Exp26] Loaded {MODEL_NAME}  |  lr={lr}")

    def train(self, dataset: GenerativeS3Dataset, loader: DataLoader, epochs: int = 3, accum_steps: int = 8):
        """
        SFT training loop.

        For each micro-batch step:
          1. Sample 1 gold + up to (max_cands-1) random negatives per question.
          2. Randomly shuffle the combined candidate list (positional debiasing).
          3. Build the MC prompt and record the gold entity name as the target.
          4. Tokenise prompts → enc (source) and targets → lbl.
          5. Replace padding token IDs with -100 in labels so T5's cross-entropy
             loss ignores padded positions.
          6. Run T5 forward (autocast bfloat16) and accumulate loss.
          7. Step optimiser every accum_steps micro-batches.

        Parameters
        ----------
        dataset     : GenerativeS3Dataset
        loader      : DataLoader (batch_size=2, shuffle=True)
        epochs      : int   — number of full passes over training data
        accum_steps : int   — gradient accumulation steps
        """
        print(f"\n[Exp26 S3] Training  |  epochs={epochs}  accum={accum_steps}")

        for ep in range(epochs):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                prompts, targets = [], []

                for item in batch:
                    q         = str(item["question"])
                    item_path = item.get("path") or ""
                    golds = [c for c in item["candidates"] if     c["is_gold"]]
                    negs  = [c for c in item["candidates"] if not c["is_gold"]]
                    if not golds: continue

                    # Randomly shuffle candidates, ensuring gold is among the top 15.
                    # This prevents the model from learning a positional shortcut
                    # (e.g., "gold is always listed first").
                    cands = golds[:1] + random.sample(negs, min(dataset.max_cands - 1, len(negs)))
                    random.shuffle(cands)

                    # Target is the exact entity name — used for teacher-forcing
                    gold_name = next(c["name"] for c in cands if c["is_gold"])

                    prompts.append(build_prompt(q, cands, item_path))
                    targets.append(gold_name)

                if not prompts:
                    continue

                # ── Tokenise source (prompt) and target (gold name) ───────────
                enc = self.tok(prompts, padding=True, truncation=True, max_length=self.max_src_len, return_tensors="pt").to(self.device)
                lbl = self.tok(targets, padding=True, truncation=True, max_length=self.max_tgt_len, return_tensors="pt").to(self.device)

                # T5 expects -100 for pad tokens in labels so they are excluded
                # from the cross-entropy loss computation.
                labels = lbl.input_ids.clone()
                labels[labels == self.tok.pad_token_id] = -100

                # ── Forward pass (bfloat16 for numerical stability with T5) ───
                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
                    # outputs.loss is the mean cross-entropy over non-padding positions
                    loss = outputs.loss / accum_steps

                loss.backward()
                if (step + 1) % accum_steps == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                total_loss += loss.item() * accum_steps
                n_batches  += 1
                pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

            avg = total_loss / max(n_batches, 1)
            print(f"  Ep{ep+1} avg_loss: {avg:.4f}")

        # ── Save final SFT checkpoint ─────────────────────────────────────────
        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"[Exp26] Final checkpoint -> {final_ckpt}")
        return final_ckpt

# ──────────────────────────────────────────────────────────────────────────────
# Main — CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp26] Device: {device}")

    train_file = os.path.join(ROOT, TRAIN_FILE)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"[Exp26] Training data not found: {train_file}")

    trainer  = Exp26Trainer(device)
    train_ds = GenerativeS3Dataset(train_file, max_cands=15)

    # Use batch_size 2 to avoid OOM with large T5 model
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_passthrough, pin_memory=True)

    trainer.train(train_ds, train_loader, epochs=3, accum_steps=8)

if __name__ == "__main__":
    main()
