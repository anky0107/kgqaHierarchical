"""
Exp 31: T5 Multiple-Choice Reranker Training
=============================================

Fine-tunes FLAN-T5-base on the hardest 50 negatives.
Objective: Given a question and 50 candidate entities + paths,
the model must output the exact string of the correct entity.
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

MODEL_NAME = "google/flan-t5-base"
TRAIN_FILE = os.path.join(ROOT, "data/exp30_t5_mc_train.json")
OUTPUT_DIR = os.path.join(ROOT, "checkpoints/exp31_t5_mc")
FINAL_CKPT = os.path.join(ROOT, "checkpoints/exp31_t5_mc_s3.pt")

import argparse

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, ignore existing checkpoints.")
    args = parser.parse_args()

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}. Run exp30 first.")
        
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    raw_dataset = load_dataset(TRAIN_FILE)
    print(f"Loaded {len(raw_dataset)} training examples.")

    def preprocess_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["prompt"],
            max_length=512,
            truncation=True,
            padding=False
        )

        # Tokenize targets
        labels = tokenizer(
            examples["target"],
            max_length=64,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=raw_dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=2, # Small batch size due to large context
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        report_to="none",
        # --- Memory optimizations to prevent VRAM paging ---
        gradient_checkpointing=True,   # Trades compute for ~40% less activation memory
        tf32=True,                     # Faster matmuls on Ampere/Ada GPUs at no accuracy cost
        dataloader_pin_memory=True,    # Faster CPU->GPU tensor transfers
        dataloader_num_workers=0,      # 0 = main process only (required on Windows with gradient_checkpointing)
    )
    
    if args.max_steps:
        training_args.max_steps = args.max_steps

    # --- Warm-resume: load only model weights, NOT the optimizer state ---
    # Root cause of previous slowdown: loading optimizer.pt (~1.9 GB) into near-full
    # VRAM (8 GB GPU, ~7.7 GB used) caused Windows WDDM to page-swap to RAM.
    # Fix: load only the model weights and let the optimizer restart fresh.
    if not args.no_resume and os.path.isdir(OUTPUT_DIR):
        ckpts = sorted(
            [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        if ckpts:
            latest_ckpt = os.path.join(OUTPUT_DIR, ckpts[-1])
            weight_file = os.path.join(latest_ckpt, "model.safetensors")
            if os.path.exists(weight_file):
                from safetensors.torch import load_file
                state_dict = load_file(weight_file, device="cpu")
                model.load_state_dict(state_dict, strict=False)
                print(f"[Exp31] Warm-resumed model weights from: {weight_file}")
                print(f"[Exp31] Optimizer state skipped intentionally to avoid VRAM paging.")
            else:
                print(f"[WARNING] No model.safetensors found in {latest_ckpt}, starting fresh.")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model weights to {FINAL_CKPT}...")
    torch.save(model.state_dict(), FINAL_CKPT)
    print("Training complete.")

if __name__ == "__main__":
    main()
