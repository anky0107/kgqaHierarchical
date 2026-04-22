# train/exp3_pct.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from transformers import BertTokenizer
import functools
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.encoder import QuestionEncoder

class PCTHopDataset(Dataset):
    def __init__(self, questions, relations, domains, subdomains=None):
        self.questions = questions
        self.relations = relations
        self.domains = domains
        # if subdomains are not provided, we mock them for architectural completeness
        if subdomains is None:
            self.subdomains = self.relations % 200
        else:
            self.subdomains = subdomains

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.relations[idx], self.domains[idx], self.subdomains[idx]

def collate_fn(batch, tokenizer):
    questions = [item[0] for item in batch]
    relations = torch.tensor([item[1] for item in batch], dtype=torch.long)
    domains = torch.tensor([item[2] for item in batch], dtype=torch.long)
    subdomains = torch.tensor([item[3] for item in batch], dtype=torch.long)
    
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return encoded, relations, domains, subdomains

class PCTModel(nn.Module):
    def __init__(self, num_domains=69, num_subdomains=200, num_relations=916, hidden_dim=1024, encoder_model="bert-base-uncased"):
        super().__init__()
        self.encoder = QuestionEncoder(model_name=encoder_model)
        
        self.shared_net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.subdomain_head = nn.Linear(hidden_dim, num_subdomains)
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask)
        h_shared = self.shared_net(h)
        
        dom_logits = self.domain_head(h_shared)
        sub_logits = self.subdomain_head(h_shared)
        rel_logits = self.relation_head(h_shared)
        conf_logits = self.confidence_head(h_shared).squeeze(-1)
        
        return dom_logits, sub_logits, rel_logits, conf_logits

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data_dir = "data/processed_entity"
    try:
        train_q = torch.load(os.path.join(data_dir, "train_questions_raw.pt"))
        train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))
        train_d = torch.load(os.path.join(data_dir, "train_domains.pt"))
        
        dev_q = torch.load(os.path.join(data_dir, "dev_questions_raw.pt"))
        dev_r = torch.load(os.path.join(data_dir, "dev_relations.pt"))
        dev_d = torch.load(os.path.join(data_dir, "dev_domains.pt"))
    except FileNotFoundError:
        print("Data unavailable. Run data builder first.")
        return

    train_dataset = PCTHopDataset(train_q, train_r, train_d)
    dev_dataset = PCTHopDataset(dev_q, dev_r, dev_d)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate)

    num_relations = int(torch.max(train_r).item()) + 1
    num_domains = int(torch.max(train_d).item()) + 1
    num_subdomains = 200 # mocked target sizes

    model = PCTModel(
        num_domains=num_domains,
        num_subdomains=num_subdomains,
        num_relations=num_relations
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_dev_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        total_rel_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for x, r, d, s in pbar:
            input_ids = x["input_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            r, d, s = r.to(device), d.to(device), s.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    dom_logits, sub_logits, rel_logits, conf_logits = model(input_ids, attention_mask)
                    loss_d = ce_loss(dom_logits, d)
                    loss_s = ce_loss(sub_logits, s)
                    loss_r = ce_loss(rel_logits, r)
                    with torch.no_grad():
                        preds = torch.argmax(rel_logits, dim=-1)
                        is_correct = (preds == r).float()
                    loss_c = bce_loss(conf_logits, is_correct)
                    loss = loss_d + loss_s + loss_r + loss_c
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                dom_logits, sub_logits, rel_logits, conf_logits = model(input_ids, attention_mask)
                loss_d = ce_loss(dom_logits, d)
                loss_s = ce_loss(sub_logits, s)
                loss_r = ce_loss(rel_logits, r)
                with torch.no_grad():
                    preds = torch.argmax(rel_logits, dim=-1)
                    is_correct = (preds == r).float()
                loss_c = bce_loss(conf_logits, is_correct)
                loss = loss_d + loss_s + loss_r + loss_c
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            total_rel_acc += is_correct.sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_rel_acc = total_rel_acc / len(train_dataset)
        
        model.eval()
        dev_correct = 0
        dev_conf_mse = 0
        
        pbar_dev = tqdm(dev_loader, desc=f"Epoch {epoch} Dev")
        with torch.no_grad():
            for x, r, d, s in pbar_dev:
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)
                r = r.to(device)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        _, _, rel_logits, conf_logits = model(input_ids, attention_mask)
                else:
                    _, _, rel_logits, conf_logits = model(input_ids, attention_mask)
                
                preds = torch.argmax(rel_logits, dim=-1)
                
                is_correct = (preds == r).float()
                dev_correct += is_correct.sum().item()
                
                probs = torch.sigmoid(conf_logits)
                dev_conf_mse += ((probs - is_correct)**2).sum().item()
                
        dev_rel_acc = dev_correct / len(dev_dataset)
        dev_conf_mse /= len(dev_dataset)
        
        print(f"Epoch {epoch} | Train Rel Acc: {train_rel_acc:.4f} | Dev Rel Acc: {dev_rel_acc:.4f} | Dev Conf MSE: {dev_conf_mse:.4f}")
        
        if dev_rel_acc > best_dev_acc:
            best_dev_acc = dev_rel_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "exp3_pct_best.pt"))
            print("Saved new best PCT model.")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()
