# train/exp0_flat_baseline.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from transformers import BertTokenizer
import functools
from tqdm import tqdm

# Handle potential import from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.encoder import QuestionEncoder
from shared.metrics import hits_at_k

class BERTHopDataset(Dataset):
    def __init__(self, questions, relations):
        # questions is a list of strings
        # relations is a tensor of shape [N]
        self.questions = questions
        self.relations = relations

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.relations[idx]

def collate_fn(batch, tokenizer):
    questions = [item[0] for item in batch]
    relations = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return encoded, relations

class BERTRelationClassifier(nn.Module):
    def __init__(self, encoder_model="bert-base-uncased", hidden_dim=1024, num_relations=916):
        super().__init__()
        self.encoder = QuestionEncoder(model_name=encoder_model)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_relations)
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
        train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))

        dev_q = torch.load(os.path.join(data_dir, "dev_questions_raw.pt"))
        dev_r = torch.load(os.path.join(data_dir, "dev_relations.pt"))
    except FileNotFoundError:
        print("Data files not found in", data_dir)
        print("Please run data/build_supervision_with_entities.py first.")
        return

    print("Train size:", len(train_q))
    print("Dev size:", len(dev_q))

    train_dataset = BERTHopDataset(train_q, train_r)
    dev_dataset = BERTHopDataset(dev_q, dev_r)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate)

    num_relations = int(torch.max(train_r).item()) + 1

    model = BERTRelationClassifier(
        encoder_model="bert-base-uncased",
        hidden_dim=1024,
        num_relations=num_relations
    ).to(device)

    # Fine-tuning BERT requires a smaller learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_dev_hit1 = 0
    patience = 10
    patience_counter = 0

    for epoch in range(30):

        # ---- Train ----
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for i, (x, y) in enumerate(pbar):
            input_ids = x["input_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            y = y.to(device)
            
            if i == 0:
                print(f"Batch 0 shapes - input_ids: {input_ids.shape}, y: {y.shape}")
                if torch.cuda.is_available():
                    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
                    print(f"Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

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

        train_hit1 = correct / total

        # ---- Dev ----
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

        dev_hit1 = correct / total
        dev_hit3 = correct_top3 / total

        print(f"Epoch {epoch}")
        print("Train Hit@1:", train_hit1)
        print("Dev Hit@1  :", dev_hit1)
        print("Dev Hit@3  :", dev_hit3)
        print("-" * 40)

        # ---- Early Stopping ----
        if dev_hit1 > best_dev_hit1:
            best_dev_hit1 = dev_hit1
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, "exp0_relation_flat_best.pt")
            )
            print("Saved new best flat model.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()
