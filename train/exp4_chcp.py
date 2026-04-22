# train/exp4_chcp.py

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
from shared.encoder import QuestionEncoder
from utils.sparql_parser import find_reasoning_path

class CHCPDataset(Dataset):
    def __init__(self, data_path, relation2id, max_hops=4):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        self.samples = []
        self.max_hops = max_hops
        
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
                # pad or truncate to max_hops
                if len(rel_ids) > max_hops:
                    rel_ids = rel_ids[:max_hops]
                else:
                    rel_ids = rel_ids + [0] * (max_hops - len(rel_ids))
                    
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
    
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    return encoded, torch.tensor(paths, dtype=torch.long)


class CHCPModel(nn.Module):
    def __init__(self, num_relations, hidden_dim=256, max_hops=4, encoder_model="bert-base-uncased", num_heads=4, num_layers=2):
        super().__init__()
        self.max_hops = max_hops
        
        self.q_encoder = QuestionEncoder(model_name=encoder_model)
        self.proj = nn.Linear(self.q_encoder.output_dim, hidden_dim)
        
        # hop initial embeddings
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        
        # Transformer encoder for cross-hop refinement
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.stop_head = nn.Linear(hidden_dim, 1) # simple stop predictor per hop
        
        # Learned transition probabilities (logits)
        self.transition_matrix = nn.Parameter(torch.randn(num_relations, num_relations))

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)
        
        # [B, encoder_dim]
        q_h = self.q_encoder(input_ids, attention_mask)
        q_proj = self.proj(q_h) # [B, hidden_dim]
        
        # initial parallel predictions: question + learned hop embedding
        # q_proj: [B, 1, hidden_dim], hop_embeddings: [max_hops, hidden_dim]
        init_repr = q_proj.unsqueeze(1) + self.hop_embeddings.unsqueeze(0) # [B, max_hops, hidden_dim]
        
        # cross-hop refinement
        refined_repr = self.transformer(init_repr) # [B, max_hops, hidden_dim]
        
        rel_logits = self.relation_head(refined_repr) # [B, max_hops, num_relations]
        stop_logits = self.stop_head(refined_repr).squeeze(-1) # [B, max_hops]
        
        return rel_logits, stop_logits


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
    max_hops = 4

    print("Loading datasets...")
    train_dataset = CHCPDataset("data/cwq_train.json", relation2id, max_hops=max_hops)
    dev_dataset = CHCPDataset("data/cwq_dev.json", relation2id, max_hops=max_hops)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate)

    model = CHCPModel(num_relations=num_relations, max_hops=max_hops).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    ce_loss = nn.CrossEntropyLoss(ignore_index=0) # assuming 0 is padding relation
    bce_loss = nn.BCEWithLogitsLoss()

    best_dev_acc = 0
    patience = 10
    patience_counter = 0

    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp4_chcp.csv")
    with open(metrics_path, "w") as f:
        f.write("epoch,train_acc,dev_acc\n")

    for epoch in range(30):
        model.train()
        total_loss = 0
        total_acc = 0
        total_valid_hops = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for x, paths in pbar:
            input_ids = x["input_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            paths = paths.to(device) # [B, max_hops]
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    rel_logits, stop_logits = model(input_ids, attention_mask)
                    loss_r = ce_loss(rel_logits.view(-1, num_relations), paths.view(-1))
                    stop_targets = (paths == 0).float()
                    loss_stop = bce_loss(stop_logits, stop_targets)
                    
                    preds = torch.argmax(rel_logits, dim=-1)
                    trans_probs = F.log_softmax(model.transition_matrix, dim=-1)
                    coherence_loss = 0
                    for k in range(max_hops - 1):
                        r_k = preds[:, k]
                        r_k1 = preds[:, k+1]
                        valid_mask = (r_k != 0) & (r_k1 != 0)
                        if valid_mask.any():
                            log_p = trans_probs[r_k[valid_mask], r_k1[valid_mask]]
                            coherence_loss -= log_p.mean()
                    if coherence_loss != 0:
                        coherence_loss = coherence_loss / (max_hops - 1)
                    
                    loss = loss_r + loss_stop + 0.1 * coherence_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                rel_logits, stop_logits = model(input_ids, attention_mask)
                loss_r = ce_loss(rel_logits.view(-1, num_relations), paths.view(-1))
                stop_targets = (paths == 0).float()
                loss_stop = bce_loss(stop_logits, stop_targets)
                preds = torch.argmax(rel_logits, dim=-1)
                trans_probs = F.log_softmax(model.transition_matrix, dim=-1)
                coherence_loss = 0
                for k in range(max_hops - 1):
                    r_k = preds[:, k]
                    r_k1 = preds[:, k+1]
                    valid_mask = (r_k != 0) & (r_k1 != 0)
                    if valid_mask.any():
                        log_p = trans_probs[r_k[valid_mask], r_k1[valid_mask]]
                        coherence_loss -= log_p.mean()
                if coherence_loss != 0:
                    coherence_loss = coherence_loss / (max_hops - 1)
                loss = loss_r + loss_stop + 0.1 * coherence_loss
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            valid_mask = paths != 0
            acc = (preds[valid_mask] == paths[valid_mask]).float().sum().item()
            total_acc += acc
            total_valid_hops += valid_mask.sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc/max(1, valid_mask.sum().item()):.4f}"})
            
        train_acc = total_acc / total_valid_hops if total_valid_hops > 0 else 0

        model.eval()
        dev_loss = 0
        dev_acc = 0
        dev_valid_hops = 0
        
        pbar_dev = tqdm(dev_loader, desc=f"Epoch {epoch} Dev")
        with torch.no_grad():
            for x, paths in pbar_dev:
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)
                paths = paths.to(device)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        rel_logits, _ = model(input_ids, attention_mask)
                else:
                    rel_logits, _ = model(input_ids, attention_mask)
                
                preds = torch.argmax(rel_logits, dim=-1)
                
                valid_mask = paths != 0
                acc = (preds[valid_mask] == paths[valid_mask]).float().sum().item()
                dev_acc += acc
                dev_valid_hops += valid_mask.sum().item()
                
        avg_dev_acc = dev_acc / dev_valid_hops if dev_valid_hops > 0 else 0
        
        print(f"Epoch {epoch} | Train Rel Acc: {train_acc:.4f} | Dev Rel Acc: {avg_dev_acc:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_acc:.4f},{avg_dev_acc:.4f}\n")

        if avg_dev_acc > best_dev_acc:
            best_dev_acc = avg_dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "exp4_chcp_best.pt"))
            print("Saved new best CHCP model.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()
