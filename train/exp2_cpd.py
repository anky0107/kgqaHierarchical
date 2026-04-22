# train/exp2_cpd.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import json
from transformers import BertTokenizer
import functools
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.encoder import QuestionEncoder, PathEncoder
from utils.sparql_parser import find_reasoning_path

class PathDataset(Dataset):
    def __init__(self, data_path, relation2id):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        self.samples = []
        for x in raw_data:
            path = find_reasoning_path(x["sparql"])
            if path is None or len(path) == 0:
                continue
            
            rel_ids = []
            valid = True
            for node, rel, direct, next_node in path:
                if rel in relation2id:
                    rel_ids.append(relation2id[rel])
                else:
                    valid = False
            
            if valid:
                self.samples.append({
                    "question": x["question"],
                    "path": rel_ids
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer):
    questions = [item["question"] for item in batch]
    paths = [item["path"] for item in batch]
    max_len = max(len(p) for p in paths)
    
    padded_paths = []
    masks = []
    for p in paths:
        pad_len = max_len - len(p)
        padded_paths.append(p + [0] * pad_len)
        masks.append([1] * len(p) + [0] * pad_len)
        
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    return encoded, torch.tensor(padded_paths, dtype=torch.long), torch.tensor(masks, dtype=torch.float)

class CPDModel(nn.Module):
    def __init__(self, num_relations, relation_dim=128, hidden_dim=256, encoder_model="bert-base-uncased"):
        super().__init__()
        self.q_encoder = QuestionEncoder(model_name=encoder_model)
        self.rel_embed = nn.Embedding(num_relations, relation_dim)
        
        self.p_encoder = PathEncoder(relation_dim, hidden_dim)
        
        proj_dim = hidden_dim * 2
        self.q_proj = nn.Linear(self.q_encoder.output_dim, proj_dim)
        self.temperature = 0.07

    def encode_path(self, path_ids):
        emb = self.rel_embed(path_ids)
        path_repr = self.p_encoder(emb)
        return F.normalize(path_repr, p=2, dim=-1)

    def forward(self, input_ids, attention_mask, pos_path_ids):
        q_h = self.q_encoder(input_ids, attention_mask)
        q_repr = F.normalize(self.q_proj(q_h), p=2, dim=-1)
        
        p_repr = self.encode_path(pos_path_ids)
        
        B, L = pos_path_ids.size()
        
        with torch.no_grad():
            rel_weights = self.rel_embed.weight
            rel_norm = F.normalize(rel_weights, p=2, dim=-1)
            sim_matrix = torch.matmul(rel_norm, rel_norm.T)
        
        num_negATIVES = 4
        neg_paths_list = []
        for i in range(B):
            path = pos_path_ids[i]
            negatives = []
            for _ in range(num_negATIVES):
                swap_idx = torch.randint(0, L, (1,)).item()
                orig_rel = path[swap_idx].item()
                
                if orig_rel == 0:
                    negatives.append(path)
                    continue
                    
                topk = torch.topk(sim_matrix[orig_rel], k=6).indices
                topk = topk[topk != orig_rel]
                if len(topk) > 0:
                    swap_rel = topk[torch.randint(0, len(topk), (1,))].item()
                else:
                    swap_rel = orig_rel
                    
                new_path = path.clone()
                new_path[swap_idx] = swap_rel
                negatives.append(new_path)
            
            neg_paths_list.append(torch.stack(negatives))
            
        neg_paths = torch.stack(neg_paths_list)
        neg_paths_flat = neg_paths.view(-1, L)
        neg_repr_flat = self.encode_path(neg_paths_flat)
        neg_repr = neg_repr_flat.view(B, num_negATIVES, -1)
        
        pos_score = torch.sum(q_repr * p_repr, dim=-1) / self.temperature
        neg_scores = torch.sum(q_repr.unsqueeze(1) * neg_repr, dim=-1) / self.temperature
        
        logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
        target = torch.zeros(B, dtype=torch.long, device=q_repr.device)
        loss = F.cross_entropy(logits, target)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == target).float().mean()
        
        return loss, acc

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    relation2id_path = "data/processed_entity/relation2id.pt"
    if not os.path.exists(relation2id_path):
        print(f"relation2id map not found at {relation2id_path}")
        return
        
    relation2id = torch.load(relation2id_path)
    num_relations = len(relation2id)

    print("Loading datasets...")
    train_dataset = PathDataset("data/cwq_train.json", relation2id)
    dev_dataset = PathDataset("data/cwq_dev.json", relation2id)
    print("Train sizes:", len(train_dataset), len(dev_dataset))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate)

    model = CPDModel(
        num_relations=num_relations,
        relation_dim=128,
        hidden_dim=256,
        encoder_model="bert-base-uncased"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_dev_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        total_acc = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for x, paths, masks in pbar:
            input_ids = x["input_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            paths = paths.to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss, acc = model(input_ids, attention_mask, paths)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, acc = model(input_ids, attention_mask, paths)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        model.eval()
        dev_loss = 0
        dev_acc = 0

        pbar_dev = tqdm(dev_loader, desc=f"Epoch {epoch} Dev")
        with torch.no_grad():
            for x, paths, masks in pbar_dev:
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)
                paths = paths.to(device)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss, acc = model(input_ids, attention_mask, paths)
                else:
                    loss, acc = model(input_ids, attention_mask, paths)
                
                dev_loss += loss.item()
                dev_acc += acc.item()

        avg_dev_loss = dev_loss / len(dev_loader)
        avg_dev_acc = dev_acc / len(dev_loader)

        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} | Dev Loss: {avg_dev_loss:.4f} Acc: {avg_dev_acc:.4f}")

        if avg_dev_acc > best_dev_acc:
            best_dev_acc = avg_dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "exp2_cpd_best.pt"))
            print("Saved new best CPD model.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()
