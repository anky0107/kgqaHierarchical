"""
train_f3_dpo.py — CDS Pipeline: F3 Generative Judge, DPO Alignment (Exp-38)
=============================================================================

Paper Section: §V-E  "Cascading Dual-Stage Filtering (CDS) — F3 DPO Fine-tuning"

Purpose
-------
Applies Direct Preference Optimization (DPO) to the Flan-T5-base multiple-choice
reranker that was Supervised Fine-Tuned in Exp-26 (train_f3_sft.py).  DPO is the
final training stage of the CDS pipeline and yields the model evaluated in
Table III of the paper.

The SFT model already knows HOW to identify the correct entity from a candidate
list.  DPO sharpens this preference: it explicitly increases the relative
probability of generating the gold entity name (chosen / e⁺) vs. a hard-negative
entity name (rejected / e⁻), while staying close to the SFT policy through the
KL-penalty term controlled by β.

DPO Objective (§V-E, Eq. 4)
-----------------------------
  L_DPO(π_θ; π_ref) = -E_{(p, e⁺, e⁻) ~ D} [
      log σ(
          β · log( π_θ(e⁺ | p) / π_ref(e⁺ | p) )
        - β · log( π_θ(e⁻ | p) / π_ref(e⁻ | p) )
      )
  ]

  Terms:
    π_θ    = the DPO policy being trained (initialised from SFT checkpoint)
    π_ref  = frozen reference policy (same SFT checkpoint, never updated)
    p      = prompt (question + candidate list from build_dpo_dataset.py)
    e⁺     = chosen response (gold entity name)
    e⁻     = rejected response (hard-negative entity name)
    β      = 0.1  — KL-penalty coefficient (low β → larger departure from π_ref)
    σ      = sigmoid function

  Intuition: the loss is minimised when the policy assigns a higher log-ratio
  to the chosen response than to the rejected response relative to the reference.
  The β term prevents the policy from drifting too far from the SFT baseline —
  crucial here because the SFT model already has strong performance and we only
  want marginal preference sharpening.

Implementation notes
--------------------
• Both `model` and `ref_model` are initialised from the SAME SFT checkpoint
  (exp31_t5_mc_s3.pt).  `ref_model` is frozen by TRL automatically and used
  only to compute the reference log-probabilities π_ref(e|p).
• The TRL DPOTrainer handles the per-response log-probability computation and
  the Bradley-Terry preference loss internally.
• gradient_checkpointing=True trades compute for memory — essential for T5-base
  with 512-token prompts on a single GPU.

Pipeline position
-----------------
  [train_f3_sft.py]  exp26_t5_generative_s3.pt (SFT init)
        │
        ▼
  [build_dpo_dataset.py]  exp37_t5_dpo_train.json
        │
        ▼
  [train_f3_dpo.py]  ← THIS FILE  (DPO alignment)
        │  HuggingFace trainer checkpoint: checkpoints/exp38_t5_dpo/
        │  final state_dict:               checkpoints/exp38_t5_dpo_s3.pt
        ▼
  [Inference]  generate answer from top-50 candidates → final KGQA prediction

Inputs
------
- data/exp37_t5_dpo_train.json          : DPO preference dataset
  Each record: {"prompt": str, "chosen": str, "rejected": str}
- checkpoints/exp31_t5_mc_s3.pt         : SFT model weights (policy init + ref)

Outputs
-------
- checkpoints/exp38_t5_dpo/             : HuggingFace Trainer checkpoint directory
- checkpoints/exp38_t5_dpo_s3.pt        : final raw state_dict for inference

Key hyperparameters
-------------------
- beta                         : 0.1   (KL-penalty; low = more preference sharpening)
- learning_rate                : 1e-6  (very low — DPO is sensitive to LR)
- per_device_train_batch_size  : 1     (memory intensive; each sample has 3 sequences)
- gradient_accumulation_steps  : 16    (effective batch = 16)
- num_train_epochs             : 2     (DPO converges faster than SFT)
- max_length                   : 576   (512 prompt + 64 target)
- bf16                         : True  (bfloat16 on CUDA for stability)
- gradient_checkpointing       : True  (memory optimisation)
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ── Path constants ─────────────────────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-base"
# SFT checkpoint from Exp-31 (variant of Exp-26 with MC format)
# Used as BOTH the policy initialisation AND the frozen reference model
SFT_CKPT   = os.path.join(ROOT, "checkpoints/exp31_t5_mc_s3.pt")
TRAIN_FILE  = os.path.join(ROOT, "data/exp37_t5_dpo_train.json")
OUTPUT_DIR  = os.path.join(ROOT, "checkpoints/exp38_t5_dpo")   # HF Trainer output
FINAL_CKPT  = os.path.join(ROOT, "checkpoints/exp38_t5_dpo_s3.pt")  # raw state_dict

import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(file_path):
    """
    Load the DPO preference dataset JSON and convert to a HuggingFace Dataset.

    The HF Dataset format is required by TRL's DPOTrainer, which internally
    maps the "prompt", "chosen", and "rejected" keys to the per-sequence
    log-probability computation.

    Parameters
    ----------
    file_path : str  — path to exp37_t5_dpo_train.json

    Returns
    -------
    datasets.Dataset  — with columns: prompt, chosen, rejected
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # β controls the trade-off between preference sharpening and KL divergence
    # from the reference policy.  Lower β → more aggressive preference sharpening.
    # Paper uses β=0.1 (§V-E), which is at the conservative end of the typical
    # range [0.05, 0.5], preserving most of the SFT model's knowledge.
    parser.add_argument("--beta", type=float, default=0.1, help="The beta factor in DPO loss.")
    args = parser.parse_args()

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}. Run exp37 first.")

    # ── Load tokeniser ────────────────────────────────────────────────────────
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load policy model (π_θ) and inject SFT weights ───────────────────────
    # π_θ is the model whose weights will be UPDATED during DPO training.
    # We initialise it from the SFT checkpoint so that DPO is a fine-tuning
    # step on an already-capable base rather than a training-from-scratch regime.
    print(f"Loading Base T5 model and injecting SFT weights from {SFT_CKPT}...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))

    # ── Load reference model (π_ref) — frozen throughout DPO ─────────────────
    # π_ref provides the baseline log-probabilities that are SUBTRACTED from
    # π_θ's log-probs in the DPO objective.  Initialising π_ref from the same
    # SFT checkpoint ensures the implicit KL constraint is anchored at the SFT
    # model's distribution (log-ratio = 0 at the start of training).
    # TRL automatically freezes ref_model — no manual requires_grad_(False) needed.
    ref_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    ref_model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))

    # ── Load DPO preference dataset ───────────────────────────────────────────
    raw_dataset = load_dataset(TRAIN_FILE)
    print(f"Loaded {len(raw_dataset)} DPO triplets.")

    # ── Configure DPO training ────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=args.beta,                   # KL-penalty coefficient (default 0.1)
        eval_strategy="no",               # no eval during training (dev eval is offline)
        learning_rate=1e-6,               # Very low learning rate for RL/DPO
                                          # DPO is sensitive to LR; 1e-6 is safe for T5
        per_device_train_batch_size=1,    # Very memory intensive — each sample requires
                                          # computing log-probs for 3 sequences (prompt,
                                          # chosen, rejected) through both π_θ and π_ref
        gradient_accumulation_steps=16,   # effective batch size = 1 × 16 = 16
        weight_decay=0.01,
        save_total_limit=1,               # keep only the most recent checkpoint
        num_train_epochs=2,               # DPO usually needs fewer epochs than SFT
        bf16=torch.cuda.is_available(),   # bfloat16 for stable mixed-precision on CUDA
        logging_steps=10,
        save_steps=500,
        report_to="none",                 # disable W&B / TensorBoard reporting
        gradient_checkpointing=True,      # trade compute for memory — essential at this
                                          # batch size with T5-base + 512-token prompts
        tf32=True,                        # TF32 for faster matmuls on Ampere GPUs
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        max_length=576,                   # 512 (prompt) + 64 (target) token budget
    )

    # ── Instantiate TRL DPOTrainer ────────────────────────────────────────────
    # DPOTrainer handles:
    #   • tokenising (prompt, chosen, rejected) triples
    #   • computing log-probs under both π_θ and π_ref
    #   • computing the Bradley-Terry DPO loss:
    #       -log σ( β*(logπ_θ(e⁺|p) - logπ_ref(e⁺|p))
    #              -β*(logπ_θ(e⁻|p) - logπ_ref(e⁻|p)) )
    #   • gradient accumulation, mixed-precision, and checkpoint saving
    trainer = DPOTrainer(
        model=model,               # trainable policy π_θ
        ref_model=ref_model,       # frozen reference π_ref
        args=dpo_config,
        train_dataset=raw_dataset,
        processing_class=tokenizer,
    )

    # ── Run DPO training ──────────────────────────────────────────────────────
    print("Starting DPO training...")
    train_result = trainer.train()
    print("Training finished.")

    # ── Save final state dict for inference use ───────────────────────────────
    # The HF Trainer saves a full checkpoint to OUTPUT_DIR; we additionally
    # save the raw state_dict as a single .pt file for easy loading at inference
    # time without the HF Trainer machinery.
    print(f"Saving final state dict to {FINAL_CKPT}...")
    os.makedirs(os.path.dirname(FINAL_CKPT), exist_ok=True)
    torch.save(model.state_dict(), FINAL_CKPT)
    print("Done!")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
