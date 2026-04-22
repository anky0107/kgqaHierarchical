"""Evaluate all checkpoints on the CWQ TEST set."""
import torch, os, sys, functools, json
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from transformers import BertTokenizer
from shared.encoder import QuestionEncoder, PathEncoder
from shared.metrics import hits_at_k
from utils.sparql_parser import find_reasoning_path

class BERTHopDataset(Dataset):
    def __init__(self, questions, targets):
        self.questions = questions
        self.targets = targets
    def __len__(self): return len(self.questions)
    def __getitem__(self, idx): return self.questions[idx], self.targets[idx]

def collate_fn(batch, tokenizer):
    questions = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    encoded = tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return encoded, targets

def eval_classifier(model_class, ckpt_name, data_q, data_t, num_classes, label, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_relations=num_classes, **kwargs).to(device) if 'num_relations' in model_class.__init__.__code__.co_varnames else model_class(num_domains=num_classes, **kwargs).to(device)
    ckpt = os.path.join(ROOT, "checkpoints", ckpt_name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)
    loader = DataLoader(BERTHopDataset(data_q, data_t), batch_size=32, collate_fn=collate)
    
    c1 = c3 = total = 0
    with torch.no_grad():
        for x, y in loader:
            ids = x["input_ids"].to(device)
            mask = x["attention_mask"].to(device)
            y = y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
            c1 += hits_at_k(logits, y, k=1)
            c3 += hits_at_k(logits, y, k=3)
            total += y.size(0)
    h1, h3 = c1/total, c3/total
    print(f"{label} | Test Hit@1: {h1:.4f} | Test Hit@3: {h3:.4f}")
    return h1, h3

def eval_exp0():
    data_dir = os.path.join(ROOT, "data/processed_entity")
    test_q = torch.load(os.path.join(data_dir, "test_questions_raw.pt"))
    test_r = torch.load(os.path.join(data_dir, "test_relations.pt"))
    train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))
    num_rel = int(torch.max(train_r).item()) + 1
    from train.exp0_flat_baseline import BERTRelationClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTRelationClassifier(num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp0_relation_flat_best.pt"), map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)
    loader = DataLoader(BERTHopDataset(test_q, test_r), batch_size=32, collate_fn=collate)
    c1 = c3 = total = 0
    with torch.no_grad():
        for x, y in loader:
            ids, mask, y = x["input_ids"].to(device), x["attention_mask"].to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
            c1 += hits_at_k(logits, y, k=1); c3 += hits_at_k(logits, y, k=3); total += y.size(0)
    h1, h3 = c1/total, c3/total
    print(f"Exp 0 | Test Hit@1: {h1:.4f} | Test Hit@3: {h3:.4f}")
    return h1, h3

def eval_exp1():
    data_dir = os.path.join(ROOT, "data/processed_entity")
    test_q = torch.load(os.path.join(data_dir, "test_questions_raw.pt"))
    test_d = torch.load(os.path.join(data_dir, "test_domains.pt"))
    train_d = torch.load(os.path.join(data_dir, "train_domains.pt"))
    num_dom = int(torch.max(train_d).item()) + 1
    from train.exp1_domain_baseline import BERTDomainClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTDomainClassifier(num_domains=num_dom).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp1_domain_best.pt"), map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)
    loader = DataLoader(BERTHopDataset(test_q, test_d), batch_size=32, collate_fn=collate)
    c1 = c3 = total = 0
    with torch.no_grad():
        for x, y in loader:
            ids, mask, y = x["input_ids"].to(device), x["attention_mask"].to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
            c1 += hits_at_k(logits, y, k=1); c3 += hits_at_k(logits, y, k=3); total += y.size(0)
    h1, h3 = c1/total, c3/total
    print(f"Exp 1 | Test Hit@1: {h1:.4f} | Test Hit@3: {h3:.4f}")
    return h1, h3

def eval_exp2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation2id = torch.load(os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    num_rel = len(relation2id)
    
    from train.exp2_cpd import CPDModel, PathDataset, collate_fn as cpd_collate
    model = CPDModel(num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp2_cpd_best.pt"), map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = PathDataset(os.path.join(ROOT, "data/cwq_test.json"), relation2id)
    collate = functools.partial(cpd_collate, tokenizer=tokenizer)
    loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate)
    
    total_acc = 0
    count = 0
    with torch.no_grad():
        for x, paths, masks in loader:
            ids, mask = x["input_ids"].to(device), x["attention_mask"].to(device)
            paths = paths.to(device)
            with torch.amp.autocast('cuda'):
                loss, acc = model(ids, mask, paths)
            total_acc += acc.item()
            count += 1
    avg_acc = total_acc / count if count > 0 else 0
    print(f"Exp 2 | Test Contrastive Acc: {avg_acc:.4f}")
    return avg_acc, "-"

def eval_exp3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(ROOT, "data/processed_entity")
    test_q = torch.load(os.path.join(data_dir, "test_questions_raw.pt"))
    test_r = torch.load(os.path.join(data_dir, "test_relations.pt"))
    train_r = torch.load(os.path.join(data_dir, "train_relations.pt"))
    train_d = torch.load(os.path.join(data_dir, "train_domains.pt"))
    num_rel = int(torch.max(train_r).item()) + 1
    num_dom = int(torch.max(train_d).item()) + 1
    
    from train.exp3_pct import PCTModel
    model = PCTModel(num_domains=num_dom, num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp3_pct_best.pt"), map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate = functools.partial(collate_fn, tokenizer=tokenizer)
    loader = DataLoader(BERTHopDataset(test_q, test_r), batch_size=32, collate_fn=collate)
    
    c1 = c3 = total = 0
    with torch.no_grad():
        for x, y in loader:
            ids, mask, y = x["input_ids"].to(device), x["attention_mask"].to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                _, _, rel_logits, _ = model(ids, mask)
            c1 += hits_at_k(rel_logits, y, k=1)
            c3 += hits_at_k(rel_logits, y, k=3)
            total += y.size(0)
    h1, h3 = c1/total, c3/total
    print(f"Exp 3 | Test Hit@1: {h1:.4f} | Test Hit@3: {h3:.4f}")
    return h1, h3

def eval_exp4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation2id = torch.load(os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    num_rel = len(relation2id)
    max_hops = 4
    
    from train.exp4_chcp import CHCPModel, CHCPDataset, collate_fn as chcp_collate
    model = CHCPModel(num_relations=num_rel, max_hops=max_hops).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp4_chcp_best.pt"), map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = CHCPDataset(os.path.join(ROOT, "data/cwq_test.json"), relation2id, max_hops=max_hops)
    collate = functools.partial(chcp_collate, tokenizer=tokenizer)
    loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate)
    
    correct = valid_hops = 0
    correct3 = 0
    with torch.no_grad():
        for x, paths in loader:
            ids, mask = x["input_ids"].to(device), x["attention_mask"].to(device)
            paths = paths.to(device)
            with torch.amp.autocast('cuda'):
                rel_logits, _ = model(ids, mask)
            preds = torch.argmax(rel_logits, dim=-1)
            vm = paths != 0
            correct += (preds[vm] == paths[vm]).float().sum().item()
            # top-3 per hop
            _, top3 = torch.topk(rel_logits, k=3, dim=-1)
            for h in range(max_hops):
                hm = paths[:, h] != 0
                if hm.any():
                    correct3 += (top3[:, h, :][hm] == paths[:, h][hm].unsqueeze(1)).any(dim=1).float().sum().item()
            valid_hops += vm.sum().item()
    h1 = correct / valid_hops if valid_hops > 0 else 0
    h3 = correct3 / valid_hops if valid_hops > 0 else 0
    print(f"Exp 4 | Test Hit@1: {h1:.4f} | Test Hit@3: {h3:.4f}")
    return h1, h3

def main():
    print("=" * 60)
    print("  EVALUATING ALL MODELS ON CWQ TEST SET")
    print("=" * 60 + "\n")
    
    results = {}
    results["exp0"] = eval_exp0()
    results["exp1"] = eval_exp1()
    results["exp2"] = eval_exp2()
    results["exp3"] = eval_exp3()
    results["exp4"] = eval_exp4()
    
    # Write results
    path = os.path.join(ROOT, "results.md")
    with open(path, "w") as f:
        f.write("# KGQA Research Experiment Results\n\n")
        f.write("## Test Set Results (CWQ Test)\n\n")
        f.write("| Experiment | Model Description | Test Hit@1 | Test Hit@3 |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Exp 0** | Flat BERT Baseline | {results['exp0'][0]:.4f} | {results['exp0'][1]:.4f} |\n")
        f.write(f"| **Exp 1** | Domain-Restricted Search | {results['exp1'][0]:.4f} | {results['exp1'][1]:.4f} |\n")
        f.write(f"| **Exp 2** | Contrastive Path Discrimination | {results['exp2'][0]:.4f} | N/A (contrastive) |\n")
        f.write(f"| **Exp 3** | Progressive Constraint Tightening | {results['exp3'][0]:.4f} | {results['exp3'][1]:.4f} |\n")
        f.write(f"| **Exp 4** | Cross-Hop Coherence Planning | {results['exp4'][0]:.4f} | {results['exp4'][1]:.4f} |\n")
        f.write(f"| **Exp 5** | RL Meta-Constraint Policy | PPO mock | N/A |\n")
        f.write("\n---\n\n## Dev Set Results (CWQ Dev)\n\n")
        f.write("| Experiment | Model Description | Dev Hit@1 | Dev Hit@3 |\n")
        f.write("|---|---|---|---|\n")
        f.write("| **Exp 0** | Flat BERT Baseline | 0.4072 | 0.8241 |\n")
        f.write("| **Exp 1** | Domain-Restricted Search | 0.7104 | 0.9689 |\n")
        f.write("| **Exp 2** | Contrastive Path Discrimination | 0.9648 | N/A (contrastive) |\n")
        f.write("| **Exp 3** | Progressive Constraint Tightening | 0.4056 | - |\n")
        f.write("| **Exp 4** | Cross-Hop Coherence Planning | 0.7358 | - |\n")
        f.write("| **Exp 5** | RL Meta-Constraint Policy | PPO Loss: 6.67 | N/A |\n")
        f.write("\n---\n\n## Performance Notes\n\n")
        f.write("- **GPU**: RTX 5070 Laptop (SM 12.0 / Blackwell)\n")
        f.write("- **PyTorch**: 2.11.0+cu128 with Mixed Precision (AMP)\n")
        f.write("- **Dataset**: ComplexWebQuestions (CWQ) 1.1\n")
        f.write("- **Training**: AdamW lr=2e-5, early stopping patience=3\n")
        f.write("\n## Metric Descriptions\n\n")
        f.write("- **Hit@1**: fraction of samples where the top-1 prediction is correct\n")
        f.write("- **Hit@3**: fraction of samples where the correct answer is in the top-3 predictions\n")
        f.write("- **Contrastive Acc (Exp 2)**: accuracy of ranking the gold path above hard negatives\n")
        f.write("- **Hit@1/3 for Exp 4**: per-hop relation prediction accuracy across multi-hop paths\n")
    
    print(f"\nResults written to {path}")

if __name__ == "__main__":
    main()
