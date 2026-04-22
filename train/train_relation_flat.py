# train/train_relation_flat.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os


# ----------------------------
# Dataset
# ----------------------------

class HopDataset(Dataset):
    def __init__(self, questions, relations):
        self.questions = questions
        self.relations = relations

    def __len__(self):
        return self.questions.size(0)

    def __getitem__(self, idx):
        return self.questions[idx], self.relations[idx]


# ----------------------------
# Model
# ----------------------------

class RelationClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_relations)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Training
# ----------------------------

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data/processed"
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_q = torch.load(os.path.join(data_dir, "train_questions.pt"))
    train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))

    dev_q = torch.load(os.path.join(data_dir, "dev_questions.pt"))
    dev_r = torch.load(os.path.join(data_dir, "dev_relations.pt"))

    print("Train size:", train_q.size())
    print("Dev size:", dev_q.size())

    train_dataset = HopDataset(train_q, train_r)
    dev_dataset = HopDataset(dev_q, dev_r)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=256)

    input_dim = train_q.size(1)
    num_relations = int(torch.max(train_r).item()) + 1

    model = RelationClassifier(
        input_dim=input_dim,
        hidden_dim=1024,
        num_relations=num_relations
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    best_dev_hit1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(20):

        # ---- Train ----
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_hit1 = correct / total

        # ---- Dev ----
        model.eval()
        correct = 0
        correct_top3 = 0
        total = 0

        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                # Hit@1
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()

                # Hit@3
                top3 = torch.topk(logits, k=3, dim=1).indices
                correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()

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
                os.path.join(ckpt_dir, "relation_flat_best.pt")
            )

            print("Saved new best flat model.")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    train()