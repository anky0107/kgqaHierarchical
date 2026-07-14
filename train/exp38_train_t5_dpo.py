"""
Exp 38: Train T5 Reranker with DPO
===================================

Applies Direct Preference Optimization (DPO) to the T5 Multiple-Choice Reranker.
We load the SFT checkpoint (exp31_t5_mc_s3.pt) and fine-tune it to prefer
the `chosen` gold answer over the `rejected` hard negative.
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

MODEL_NAME = "google/flan-t5-base"
SFT_CKPT = os.path.join(ROOT, "checkpoints/exp31_t5_mc_s3.pt")
TRAIN_FILE = os.path.join(ROOT, "data/exp37_t5_dpo_train.json")
OUTPUT_DIR = os.path.join(ROOT, "checkpoints/exp38_t5_dpo")
FINAL_CKPT = os.path.join(ROOT, "checkpoints/exp38_t5_dpo_s3.pt")

import argparse

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1, help="The beta factor in DPO loss.")
    args = parser.parse_args()

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}. Run exp37 first.")
        
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading Base T5 model and injecting SFT weights from {SFT_CKPT}...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))
    
    ref_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    ref_model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))

    raw_dataset = load_dataset(TRAIN_FILE)
    print(f"Loaded {len(raw_dataset)} DPO triplets.")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=args.beta,
        eval_strategy="no",
        learning_rate=1e-6,           # Very low learning rate for RL/DPO
        per_device_train_batch_size=1, # Very memory intensive, keep batch size small
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=2,            # DPO usually needs fewer epochs
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        report_to="none",
        gradient_checkpointing=True,
        tf32=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        max_length=576,               # 512 + 64 (target)
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=raw_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    train_result = trainer.train()
    print("Training finished.")

    print(f"Saving final state dict to {FINAL_CKPT}...")
    os.makedirs(os.path.dirname(FINAL_CKPT), exist_ok=True)
    torch.save(model.state_dict(), FINAL_CKPT)
    print("Done!")

if __name__ == "__main__":
    main()
