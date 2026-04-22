# # train/train_relation_hierarchical.py

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import os


# # ----------------------------
# # Dataset
# # ----------------------------

# class HopDataset(Dataset):
#     def __init__(self, questions, relations, domains):
#         self.questions = questions
#         self.relations = relations
#         self.domains = domains

#     def __len__(self):
#         return self.questions.size(0)

#     def __getitem__(self, idx):
#         return (
#             self.questions[idx],
#             self.relations[idx],
#             self.domains[idx]
#         )


# # ----------------------------
# # Model
# # ----------------------------

# class HierarchicalRelationModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_relations, num_domains):
#         super().__init__()

#         self.shared = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim)
#         )

#         self.domain_head = nn.Linear(hidden_dim, num_domains)
#         self.relation_head = nn.Linear(hidden_dim, num_relations)

#     def forward(self, x):
#         h = self.shared(x)
#         z_d = self.domain_head(h)
#         z_r_base = self.relation_head(h)
#         return z_r_base, z_d


# # ----------------------------
# # Training
# # ----------------------------

# def train():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     data_dir = "data/processed"
#     ckpt_dir = "checkpoints"
#     os.makedirs(ckpt_dir, exist_ok=True)

#     # Load tensors
#     train_q = torch.load(os.path.join(data_dir, "train_questions.pt"))
#     train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))
#     train_d = torch.load(os.path.join(data_dir, "train_domains.pt"))

#     dev_q = torch.load(os.path.join(data_dir, "dev_questions.pt"))
#     dev_r = torch.load(os.path.join(data_dir, "dev_relations.pt"))
#     dev_d = torch.load(os.path.join(data_dir, "dev_domains.pt"))

#     relation_to_domain = torch.load(
#         os.path.join(data_dir, "relation_to_domain.pt")
#     ).to(device)

#     num_relations = relation_to_domain.size(0)
#     num_domains = int(torch.max(train_d).item()) + 1

#     train_dataset = HopDataset(train_q, train_r, train_d)
#     dev_dataset = HopDataset(dev_q, dev_r, dev_d)

#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#     dev_loader = DataLoader(dev_dataset, batch_size=256)

#     model = HierarchicalRelationModel(
#         input_dim=train_q.size(1),
#         hidden_dim=1024,
#         num_relations=num_relations,
#         num_domains=num_domains
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
#     ce = nn.CrossEntropyLoss()

#     alpha = 1.0
#     beta = 0.5

#     best_dev_hit1 = 0
#     patience = 3
#     patience_counter = 0

#     for epoch in range(20):

#         # ---- Train ----
#         model.train()
#         total = 0
#         correct = 0

#         for x, r_true, d_true in train_loader:

#             x = x.to(device)
#             r_true = r_true.to(device)
#             d_true = d_true.to(device)

#             z_r_base, z_d = model(x)

#             # Domain-conditioned relation logits
#             domain_bias = z_d[:, relation_to_domain]  # [B, num_relations]
#             z_r = z_r_base + alpha * domain_bias

#             loss_relation = ce(z_r, r_true)
#             loss_domain = ce(z_d, d_true)

#             loss = loss_relation + beta * loss_domain

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             preds = torch.argmax(z_r, dim=1)
#             correct += (preds == r_true).sum().item()
#             total += r_true.size(0)

#         train_hit1 = correct / total

#         # ---- Dev ----
#         model.eval()
#         correct = 0
#         correct_top3 = 0
#         total = 0

#         with torch.no_grad():
#             for x, r_true, d_true in dev_loader:

#                 x = x.to(device)
#                 r_true = r_true.to(device)

#                 z_r_base, z_d = model(x)

#                 domain_bias = z_d[:, relation_to_domain]
#                 z_r = z_r_base + alpha * domain_bias

#                 preds = torch.argmax(z_r, dim=1)
#                 correct += (preds == r_true).sum().item()

#                 top3 = torch.topk(z_r, k=3, dim=1).indices
#                 correct_top3 += (
#                     (top3 == r_true.unsqueeze(1))
#                     .any(dim=1)
#                     .sum()
#                     .item()
#                 )

#                 total += r_true.size(0)

#         dev_hit1 = correct / total
#         dev_hit3 = correct_top3 / total

#         print(f"Epoch {epoch}")
#         print("Train Hit@1:", train_hit1)
#         print("Dev Hit@1  :", dev_hit1)
#         print("Dev Hit@3  :", dev_hit3)
#         print("-" * 40)

#         # Early stopping
#         if dev_hit1 > best_dev_hit1:
#             best_dev_hit1 = dev_hit1
#             patience_counter = 0
#             torch.save(
#                 model.state_dict(),
#                 os.path.join(ckpt_dir, "relation_hierarchical_best.pt")
#             )
#             print("Saved new best hierarchical model.")
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break


# if __name__ == "__main__":
#     train()


# train/train_relation_hierarchical_entity.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os


# ----------------------------
# Dataset
# ----------------------------

class HopDataset(Dataset):
    def __init__(self, questions, entities, relations, domains):
        self.questions = questions
        self.entities = entities
        self.relations = relations
        self.domains = domains

    def __len__(self):
        return self.questions.size(0)

    def __getitem__(self, idx):
        return (
            self.questions[idx],
            self.entities[idx],
            self.relations[idx],
            self.domains[idx]
        )


# ----------------------------
# Model
# ----------------------------

class HierarchicalEntityModel(nn.Module):
    def __init__(self,
                 question_dim,
                 num_entities,
                 entity_dim,
                 hidden_dim,
                 num_relations,
                 num_domains):
        super().__init__()

        self.entity_embedding = nn.Embedding(num_entities, entity_dim)

        self.shared = nn.Sequential(
            nn.Linear(question_dim + entity_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.domain_head = nn.Linear(hidden_dim, num_domains)

    def forward(self, question_emb, entity_ids):
        e_emb = self.entity_embedding(entity_ids)
        h = torch.cat([question_emb, e_emb], dim=1)
        h = self.shared(h)

        z_r_base = self.relation_head(h)
        z_d = self.domain_head(h)

        return z_r_base, z_d


# ----------------------------
# Training
# ----------------------------

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data/processed_entity"
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load tensors
    train_q = torch.load(os.path.join(data_dir, "train_questions.pt"))
    train_e = torch.load(os.path.join(data_dir, "train_entities.pt"))
    train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))
    train_d = torch.load(os.path.join(data_dir, "train_domains.pt"))

    dev_q = torch.load(os.path.join(data_dir, "dev_questions.pt"))
    dev_e = torch.load(os.path.join(data_dir, "dev_entities.pt"))
    dev_r = torch.load(os.path.join(data_dir, "dev_relations.pt"))
    dev_d = torch.load(os.path.join(data_dir, "dev_domains.pt"))

    relation_to_domain = torch.load(
        os.path.join(data_dir, "relation_to_domain.pt")
    ).to(device)

    entity2id = torch.load(os.path.join(data_dir, "entity2id.pt"))

    num_entities = len(entity2id)
    num_relations = relation_to_domain.size(0)
    num_domains = int(torch.max(train_d).item()) + 1

    print("Entities:", num_entities)
    print("Relations:", num_relations)
    print("Domains:", num_domains)

    train_dataset = HopDataset(train_q, train_e, train_r, train_d)
    dev_dataset = HopDataset(dev_q, dev_e, dev_r, dev_d)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=256)

    model = HierarchicalEntityModel(
        question_dim=train_q.size(1),
        num_entities=num_entities,
        entity_dim=256,
        hidden_dim=1024,
        num_relations=num_relations,
        num_domains=num_domains
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ce = nn.CrossEntropyLoss()

    alpha = 1.0
    beta = 0.5

    best_dev_hit1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(20):

        # ---- Train ----
        model.train()
        correct = 0
        total = 0

        for q, e, r, d in train_loader:

            q = q.to(device)
            e = e.to(device)
            r = r.to(device)
            d = d.to(device)

            z_r_base, z_d = model(q, e)

            # Domain-conditioned logits
            domain_bias = z_d[:, relation_to_domain]
            z_r = z_r_base + alpha * domain_bias

            loss_relation = ce(z_r, r)
            loss_domain = ce(z_d, d)
            loss = loss_relation + beta * loss_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(z_r, dim=1)
            correct += (preds == r).sum().item()
            total += r.size(0)

        train_hit1 = correct / total

        # ---- Dev ----
        model.eval()
        correct = 0
        correct_top3 = 0
        total = 0

        with torch.no_grad():
            for q, e, r, d in dev_loader:

                q = q.to(device)
                e = e.to(device)
                r = r.to(device)

                z_r_base, z_d = model(q, e)
                domain_bias = z_d[:, relation_to_domain]
                z_r = z_r_base + alpha * domain_bias

                preds = torch.argmax(z_r, dim=1)
                correct += (preds == r).sum().item()

                top3 = torch.topk(z_r, k=3, dim=1).indices
                correct_top3 += (
                    (top3 == r.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )

                total += r.size(0)

        dev_hit1 = correct / total
        dev_hit3 = correct_top3 / total

        print(f"Epoch {epoch}")
        print("Train Hit@1:", train_hit1)
        print("Dev Hit@1  :", dev_hit1)
        print("Dev Hit@3  :", dev_hit3)
        print("-" * 40)

        if dev_hit1 > best_dev_hit1:
            best_dev_hit1 = dev_hit1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, "relation_hierarchical_entity_best.pt")
            )

            print("Saved new best hierarchical entity model.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    train()