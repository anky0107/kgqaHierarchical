"""
Exp 28: Pure Feature-Based RL Reranker (Stage 3)
================================================

Replaces Stage 3 PLMs entirely with a lightweight MLP Policy.
Uses extracted textual/graph features and trains via REINFORCE.

Checkpoint: checkpoints/exp28_pure_rl_s3.pt
"""

import os, sys, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

from cds_pipeline.utils import path_to_nl
from cds_pipeline.rl_features import extract_features
from cds_pipeline.models import PureRLPolicy

CKPT_NAME   = "exp28_pure_rl_s3.pt"
TRAIN_FILE  = "data/exp18_cds_train_hard_full.json"
DEV_FILE    = "data/exp16_cds_dev.json"

# ── Dataset ───────────────────────────────────────────────────────────────────

class PureRLDataset(Dataset):
    def __init__(self, json_path: str, max_cands: int = 15):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.max_cands = max_cands
        self.samples = [s for s in raw if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp28 Dataset] {len(self.samples)} samples from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

def collate_passthrough(batch):
    return batch

# ── Trainer ───────────────────────────────────────────────────────────────────

class Exp28Trainer:
    def __init__(self, device: torch.device, lr: float = 1e-3):
        self.device = device
        self.model = PureRLPolicy().to(device)
        self.opt = AdamW(self.model.parameters(), lr=lr)
        print(f"[Exp28] Loaded PureRLPolicy (Feature-Based)  |  lr={lr}")

    def train(self, dataset: PureRLDataset, loader: DataLoader, epochs: int = 10, accum_steps: int = 16, entropy_coef: float = 0.05):
        print(f"\n[Exp28 RL] Training REINFORCE  |  epochs={epochs}  accum={accum_steps}")
        
        baseline = 0.0  # Moving average baseline for rewards
        alpha = 0.05    # Smoothing factor for baseline

        for ep in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_reward = 0.0
            n_batches = 0
            
            self.opt.zero_grad()
            pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}")

            for step, batch in enumerate(pbar):
                loss = 0.0
                batch_reward = 0.0
                valid_items = 0

                for item in batch:
                    q = str(item["question"])
                    item_path = item.get("path") or ""
                    golds = [c for c in item["candidates"] if c["is_gold"]]
                    negs  = [c for c in item["candidates"] if not c["is_gold"]]
                    if not golds: continue
                    
                    cands = golds[:1] + random.sample(negs, min(dataset.max_cands - 1, len(negs)))
                    random.shuffle(cands)
                    
                    gold_idx = next(i for i, c in enumerate(cands) if c["is_gold"])
                    
                    feature_matrix = []
                    for c in cands:
                        cand_path_str = c.get("path") or item_path
                        path_nl = path_to_nl(cand_path_str)
                        name = c.get("name", "")
                        feat = extract_features(q, name, path_nl)
                        feature_matrix.append(feat)

                    feat_tensor = torch.tensor(feature_matrix, dtype=torch.float32, device=self.device)

                    # [K] logits
                    logits = self.model(feat_tensor)
                    
                    # Policy Distribution
                    probs = F.softmax(logits, dim=0)
                    log_probs = F.log_softmax(logits, dim=0)
                    
                    # Entropy Regularization
                    entropy = -torch.sum(probs * log_probs)
                    
                    # Sample action
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    
                    # Calculate Reward
                    reward = 1.0 if action.item() == gold_idx else 0.0
                    batch_reward += reward
                    
                    # Update baseline
                    baseline = (1 - alpha) * baseline + alpha * reward
                    
                    # REINFORCE loss
                    advantage = reward - baseline
                    policy_loss = -advantage * log_probs[action]
                    
                    item_loss = policy_loss - (entropy_coef * entropy)
                    loss = loss + item_loss
                    valid_items += 1

                if valid_items == 0:
                    continue

                loss = loss / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                total_loss += loss.item() * accum_steps
                total_reward += batch_reward / valid_items
                n_batches += 1
                
                pbar.set_postfix(
                    loss=f"{total_loss/n_batches:.4f}", 
                    rew=f"{total_reward/n_batches:.3f}", 
                    adv=f"{advantage:.3f}"
                )

            print(f"  Ep{ep+1} avg_reward: {total_reward/max(n_batches,1):.3f}")
            
            ep_ckpt = os.path.join(ROOT, "checkpoints", f"exp28_rl_epoch{ep+1}.pt")
            torch.save(self.model.state_dict(), ep_ckpt)

        final_ckpt = os.path.join(ROOT, "checkpoints", CKPT_NAME)
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"[Exp28] Final checkpoint -> {final_ckpt}")
        return final_ckpt

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp28] Device: {device}")

    train_file = os.path.join(ROOT, TRAIN_FILE)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"[Exp28] Training data not found: {train_file}")

    trainer = Exp28Trainer(device)
    train_ds = PureRLDataset(train_file, max_cands=15)
    
    # Very fast, can use large batch
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_passthrough, pin_memory=True)

    trainer.train(train_ds, train_loader, epochs=15, accum_steps=2)

if __name__ == "__main__":
    main()
