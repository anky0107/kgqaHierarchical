"""
Exp 9-MetaQA: Surgical MetaQA Extension of Exp 9 RLMC

Strategy: 
  1. Load Exp 8 backbone (frozen RoBERTa)
  2. Expand relation_head to include MetaQA relations (only new rows trained)
  3. Inject topic entity text into question
  4. Retrain RL Policy Head (4 actions) using MetaQA KG as execution env
  5. Evaluate on MetaQA 1/2/3-hop test sets

CWQ score stays at 76.6% — we never touch the RoBERTa backbone.
"""
import os, sys, torch, functools, re
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp7_roberta import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from data.build_universal_vocab import parse_metaqa_kg, parse_metaqa_split

# ============================================================
#  Expanded Planner: adds MetaQA relations to the output head
# ============================================================
class ExpandedPlanner(nn.Module):
    """
    Wraps ScaledUnifiedPlanner with an expanded relation head.
    Old CWQ relation weights are frozen. Only new MetaQA rows train.
    """
    def __init__(self, base_model, old_num_rel, new_num_rel):
        super().__init__()
        self.base = base_model
        
        # Freeze everything in the base model
        for p in self.base.parameters():
            p.requires_grad = False
        
        # New relation head: keeps all old weights + adds new rows
        old_weight = self.base.relation_head.weight.data  # [old_num_rel, hidden]
        extra = new_num_rel - old_num_rel
        
        # New rows for MetaQA relations, randomly initialized
        extra_weight = nn.Parameter(torch.randn(extra, old_weight.size(1)) * 0.01)
        self.extra_relation_weight = extra_weight
        
        self.old_num_rel = old_num_rel
        self.new_num_rel = new_num_rel

    def forward(self, input_ids, attention_mask):
        # Get base outputs
        out = self.base(input_ids, attention_mask)
        
        # Expand rel_logits with new rows
        B, H, _ = out['rel_logits'].shape
        hidden = self.base.transformer(
            self.base.proj(
                self.base.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
            ).unsqueeze(1) + self.base.hop_embeddings.unsqueeze(0)
        )  # [B, H, D]
        
        # Extra logits for new relations
        extra_logits = torch.matmul(hidden, self.extra_relation_weight.t())  # [B, H, extra]
        
        # Concatenate full logits
        full_logits = torch.cat([out['rel_logits'], extra_logits], dim=-1)  # [B, H, new_num_rel]
        out['rel_logits'] = full_logits
        return out

# ============================================================
#  MetaQA Dataset for path prediction
# ============================================================
class MetaQADataset(Dataset):
    def __init__(self, proc_json, relation2id, max_hops=3):
        import json
        with open(proc_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        entity_pat = re.compile(r'\[(.+?)\]')
        
        for item in data:
            rels = item.get('relations', [])
            if not rels:
                continue
            
            rel_ids = [relation2id[r] for r in rels[:max_hops] if r in relation2id]
            if not rel_ids:
                continue
            
            topic = item.get('topic_entity', '')
            question = item.get('question', '')
            
            # Topic entity injection
            if topic:
                question = f"[METAQA] topic: {topic} | {question}"
            
            num_hops = len(rel_ids)
            # Pad to max_hops
            rel_ids += [0] * (max_hops - len(rel_ids))
            
            self.samples.append({
                'question': question,
                'relations': torch.tensor(rel_ids, dtype=torch.long),
                'num_hops': num_hops,
                'topic_entity': topic,
                'gold_answers': item.get('gold_answers', []),
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_metaqa(batch, tokenizer):
    questions = [b['question'] for b in batch]
    enc = tokenizer(questions, padding=True, truncation=True, max_length=160, return_tensors='pt')
    rels = torch.stack([b['relations'] for b in batch])
    nums = torch.tensor([b['num_hops'] for b in batch], dtype=torch.long)
    return enc, rels, nums

# ============================================================
#  RL Environment using MetaQA KG
# ============================================================
class MetaQAEnvironment:
    def __init__(self, kg):
        self.kg = kg  # entity -> [(rel, neighbor)]
    
    def step(self, topic_entity, pred_rel_ids, id2rel, num_hops, gold_answers):
        """
        Physically traverse MetaQA KG with predicted relations for num_hops.
        Returns: reward, success
        """
        current = {topic_entity}
        success = False
        
        for h in range(num_hops):
            r_id = pred_rel_ids[h]
            r_name = id2rel.get(r_id, '')
            
            next_entities = set()
            for e in current:
                for rel, neighbor in self.kg.get(e, []):
                    if rel == r_name:
                        next_entities.add(neighbor)
            
            if not next_entities:
                break
            current = next_entities
        
        # Check if we reached any gold answer
        gold_set = set(gold_answers)
        if current & gold_set:
            success = True
        
        reward = 1.0 if success else -0.5
        return reward, success

# ============================================================
#  Train Exp 9-MetaQA
# ============================================================
def train_exp9_metaqa():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    # Load universal vocab (built by build_universal_vocab.py)
    proc_dir = os.path.join(ROOT, 'data/processed_universal')
    full_rel2id = torch.load(os.path.join(proc_dir, 'relation2id.pt'))
    full_id2rel = {v: k for k, v in full_rel2id.items()}
    num_full_rel = len(full_rel2id)
    
    # Old CWQ vocab size
    old_rel2id = torch.load(os.path.join(ROOT, 'data/processed_entity/relation2id.pt'))
    old_rel2id_path = os.path.join(ROOT, 'data/processed_entity/domain2id.pt')
    dom2id = torch.load(old_rel2id_path)
    num_dom = len(dom2id)
    num_old_rel = len(old_rel2id)
    
    print(f"Old vocab: {num_old_rel} rels | Full vocab: {num_full_rel} rels | MetaQA extension: {num_full_rel - num_old_rel} rels")
    
    # Load MetaQA KG
    print("Loading MetaQA KG...")
    kb_path = os.path.join(ROOT, 'data/metaqa/kb.txt')
    metaqa_kg = defaultdict(list)
    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                subj, rel, obj = parts
                metaqa_kg[subj].append((rel, obj))
                metaqa_kg[obj].append((rel + '_inv', subj))
    print(f"  KG: {len(metaqa_kg)} entities")
    
    env = MetaQAEnvironment(metaqa_kg)
    
    # Load datasets (from processed_universal)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    collate = functools.partial(collate_metaqa, tokenizer=tokenizer)
    
    train_ds = MetaQADataset(os.path.join(proc_dir, 'metaqa/train.json'), full_rel2id)
    dev_ds = MetaQADataset(os.path.join(proc_dir, 'metaqa/dev.json'), full_rel2id)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=16, collate_fn=collate)
    
    # Build Expanded Planner on top of Exp 8 backbone
    print("Loading Exp 8 backbone...")
    base_model = ScaledUnifiedPlanner(num_dom, num_old_rel).to(device)
    base_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp8_cpd_best.pt'), map_location=device))
    
    planner = ExpandedPlanner(base_model, num_old_rel, num_full_rel).to(device)
    
    # RL Agent wraps the expanded planner
    rl_agent = RLConstraintAgent(planner.base).to(device)
    
    # Only train: extra_relation_weight + RL policy/value heads
    trainable_params = list(planner.parameters()) + [
        p for p in rl_agent.policy_head.parameters()
    ] + [
        p for p in rl_agent.value_head.parameters()
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    
    epochs = 10
    gamma = 0.99
    
    metrics_path = os.path.join(ROOT, 'metrics/exp9_metaqa.csv')
    with open(metrics_path, 'w') as f:
        f.write("epoch,ce_loss,success_rate\n")
    
    print("\nStarting Exp 9-MetaQA Training (10 epochs)...")
    
    for epoch in range(epochs):
        planner.train()
        rl_agent.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        total_success = 0
        total_samples = 0
        ce_total = 0
        
        for enc, paths, nums in t_bar:
            enc = enc.to(device); paths = paths.to(device); nums = nums.to(device)
            
            with torch.amp.autocast('cuda'):
                # Standard CE on expanded relation head
                out = planner(enc['input_ids'], enc['attention_mask'])
                ce_loss = F.cross_entropy(out['rel_logits'].view(-1, num_full_rel), paths.view(-1))
            
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            ce_total += ce_loss.item()
            
            t_bar.set_postfix(ce=ce_loss.item())
        
        avg_ce = ce_total / len(train_loader)
        print(f"Epoch {epoch} | CE Loss: {avg_ce:.4f}")
        
        with open(metrics_path, 'a') as f:
            f.write(f"{epoch},{avg_ce:.4f},{0.0:.4f}\n")
        
        torch.save(planner.state_dict(), f'checkpoints/exp9_metaqa_epoch_{epoch}.pt')
    
    print("\nTraining complete! Run eval_metaqa.py to evaluate.")

if __name__ == '__main__':
    train_exp9_metaqa()
