"""
Exp 26: Generative Reranking for Stage 3 (T5)
=============================================

Replaces the Stage 3 Cross-Encoder with a Sequence-to-Sequence LLM (google/flan-t5-large).
Given the top-15 candidates from Stage 2, the LLM is prompted to output the exact name of the gold entity.

Checkpoint: checkpoints/exp26_t5_generative_s3.pt
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

MODEL_NAME  = "google/flan-t5-base"
CKPT_NAME   = "exp26_t5_generative_s3.pt"
TRAIN_FILE  = "data/exp18_cds_train_hard_full.json"
DEV_FILE    = "data/exp16_cds_dev.json"

# ── Input formatting ──────────────────────────────────────────────────────────

def build_prompt(question: str, candidates: list, item_path: str) -> str:
    prompt = f"Question: {question}\n\nCandidates:\n"
    for i, c in enumerate(candidates, 1):
        name = c.get("name", "").strip() or "[UNK]"
        path_str = c.get("path") or item_path or ""
        path_nl = path_to_nl(path_str)
        if path_nl:
            prompt += f"{i}. {name} (Path: {path_nl})\n"
        else:
            prompt += f"{i}. {name}\n"
    prompt += "\nWhich of the above candidates is the correct answer to the question? Answer with the exact name."
    return prompt

# ── Dataset ───────────────────────────────────────────────────────────────────

class GenerativeS3Dataset(Dataset):
    def __init__(self, json_path: str, max_cands: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_cands = max_cands
        self.samples = [s for s in raw if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp26 Dataset] {len(self.samples)} samples from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

def collate_passthrough(batch):
    return batch

# ── Trainer ───────────────────────────────────────────────────────────────────

class Exp26Trainer:
    def __init__(self, device: torch.device, lr: float = 1e-4, max_src_len: int = 512, max_tgt_len: int = 64):
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        self.opt = AdamW(self.model.parameters(), lr=lr)
        print(f"[Exp26] Loaded {MODEL_NAME}  |  lr={lr}")

    def train(self, dataset: GenerativeS3Dataset, loader: DataLoader, epochs: int = 3, accum_steps: int = 8):
        print(f"\n[Exp26 S3] Training  |  epochs={epochs}  accum={accum_steps}")

        for ep in range(epochs):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                prompts, targets = [], []

                for item in batch:
                    q = str(item["question"])
                    item_path = item.get("path") or ""
                    golds = [c for c in item["candidates"] if c["is_gold"]]
                    negs  = [c for c in item["candidates"] if not c["is_gold"]]
                    if not golds: continue
                    
                    # Randomly shuffle candidates, ensuring gold is among the top 15
                    cands = golds[:1] + random.sample(negs, min(dataset.max_cands - 1, len(negs)))
                    random.shuffle(cands)
                    
                    gold_name = next(c["name"] for c in cands if c["is_gold"])
                    
                    prompts.append(build_prompt(q, cands, item_path))
                    targets.append(gold_name)

                if not prompts:
                    continue

                enc = self.tok(prompts, padding=True, truncation=True, max_length=self.max_src_len, return_tensors="pt").to(self.device)
                lbl = self.tok(targets, padding=True, truncation=True, max_length=self.max_tgt_len, return_tensors="pt").to(self.device)
                
                # T5 expects -100 for pad tokens in labels
                labels = lbl.input_ids.clone()
                labels[labels == self.tok.pad_token_id] = -100

                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
                    loss = outputs.loss / accum_steps

                loss.backward()
                if (step + 1) % accum_steps == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                total_loss += loss.item() * accum_steps
                n_batches += 1
                pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

            avg = total_loss / max(n_batches, 1)
            print(f"  Ep{ep+1} avg_loss: {avg:.4f}")

        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"[Exp26] Final checkpoint -> {final_ckpt}")
        return final_ckpt

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp26] Device: {device}")

    train_file = os.path.join(ROOT, TRAIN_FILE)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"[Exp26] Training data not found: {train_file}")

    trainer = Exp26Trainer(device)
    train_ds = GenerativeS3Dataset(train_file, max_cands=15)
    
    # Use batch_size 2 to avoid OOM with large T5 model
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_passthrough, pin_memory=True)

    trainer.train(train_ds, train_loader, epochs=3, accum_steps=8)

if __name__ == "__main__":
    main()
