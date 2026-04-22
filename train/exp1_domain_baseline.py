# train/exp1_domain_baseline.py

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
from shared.metrics import hits_at_k

class BERTHopDataset(Dataset):
    def __init__(self, questions, targets):
        self.questions = questions
        self.targets = targets

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.targets[idx]

def collate_fn(batch, tokenizer):
    questions = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return encoded, targets

class BERTDomainClassifier(nn.Module):
    def __init__(self, encoder_model="bert-base-uncased", hidden_dim=1024, num_domains=69):
        super().__init__()
        self.encoder = QuestionEncoder(model_name=encoder_model)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask)
        logits = self.net(h)
        return logits

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    data_dir = "data/processed_entity"
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        train_q = torch.load(os.path.join(data_dir, "train_questions_raw.pt"))
        train_d = torch.load(os.path.join(data_dir, "train_domains.pt"))

        dev_q = torch.load(os.path.join(data_dir, "dev_questions_raw.pt"))
        dev_d = torch.load(os.path.join(data_dir, "dev_domains.pt"))
    except FileNotFoundError:
        print("Data files not found in", data_dir)
        return

    train_dataset = BERTHopDataset(train_q, train_d)
    dev_dataset = BERTHopDataset(dev_q, dev_d)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate)

    num_domains = int(torch.max(train_d).item()) + 1

    model = BERTDomainClassifier(
        encoder_model="bert-base-uncased",
        hidden_dim=1024,
        num_domains=num_domains
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_dev_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(30):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for x, y in pbar:
            input_ids = x["input_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += hits_at_k(logits, y, k=1)
            total += y.size(0)
            pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

        train_acc = correct / total

        model.eval()
        correct = 0
        correct_top3 = 0
        total = 0

        pbar_dev = tqdm(dev_loader, desc=f"Epoch {epoch} Dev")
        with torch.no_grad():
            for x, y in pbar_dev:
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)
                y = y.to(device)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask)
                else:
                    logits = model(input_ids, attention_mask)
                    
                correct += hits_at_k(logits, y, k=1)
                correct_top3 += hits_at_k(logits, y, k=3)
                total += y.size(0)

        dev_acc = correct / total
        dev_top3 = correct_top3 / total

        print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f} | Dev Top3: {dev_top3:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "exp1_domain_best.pt"))
            print("Saved new best domain model.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

class DomainRestrictedSearcher:
    def __init__(self, classifier_path, relation_to_domain_path, kg_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERTDomainClassifier().to(self.device)
        if os.path.exists(classifier_path):
            self.model.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.relation_to_domain = torch.load(relation_to_domain_path) if os.path.exists(relation_to_domain_path) else None
        self.kg = kg_loader
        
    def search(self, question_text, start_entity, beam_width=3):
        inputs = self.tokenizer(question_text, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            top_domains = torch.topk(logits, k=beam_width, dim=-1).indices[0].cpu().tolist()
            
        allowed_relations = set()
        if self.relation_to_domain is not None:
            for r_idx, d_idx in enumerate(self.relation_to_domain.tolist()):
                if d_idx in top_domains:
                    allowed_relations.add(r_idx)
                    
        beam = [(start_entity, [])]
        for hop in range(2):
            new_beam = []
            for entity, path in beam:
                neighbors = self.kg.get_neighbors(entity)
                for r, t, direction in neighbors:
                    new_beam.append((t, path + [(r, t)]))
            beam = new_beam[:beam_width]
        return beam

if __name__ == "__main__":
    train()
