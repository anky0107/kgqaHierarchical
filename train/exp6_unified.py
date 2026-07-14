"""
Exp 6: Unified Adaptive-CHCP Model

Combines:
  1. Progressive Constraint Tightening (Exp 3) -> Confidence/Stopping
  2. Cross-Hop Coherence Planning (Exp 4) -> Joint Transformer Reasoning
"""
import os, sys, json, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared.encoder import QuestionEncoder
from utils.sparql_parser import find_reasoning_path

# ============================================================
#  Model Architecture
# ============================================================

class UnifiedKGQAPlanner(nn.Module):
    def __init__(self, num_domains, num_relations, hidden_dim=256, max_hops=4):
        super().__init__()
        self.max_hops = max_hops
        
        self.q_encoder = QuestionEncoder(model_name="bert-base-uncased")
        self.proj = nn.Linear(self.q_encoder.output_dim, hidden_dim)
        
        # 1. Progressive Constraint heads (from Exp 3)
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.confidence_head = nn.Linear(hidden_dim, 1) # scalar confidence
        
        # 2. Coherent Planner (from Exp 4)
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.adaptive_stop_head = nn.Linear(hidden_dim, 1) # learned stop per hop

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)
        
        # BERT Encoding
        q_h = self.q_encoder(input_ids, attention_mask)
        h_q = self.proj(q_h) # [B, hidden_dim]
        
        # Global Domain & Confidence (Exp 3 elements)
        domain_logits = self.domain_head(h_q)
        q_confidence = torch.sigmoid(self.confidence_head(h_q))
        
        # Cross-Hop Reasoning (Exp 4 elements)
        # Combine question with learned hop positions
        init_repr = h_q.unsqueeze(1) + self.hop_embeddings.unsqueeze(0) # [B, max_hops, hidden_dim]
        refined_repr = self.transformer(init_repr) # [B, max_hops, hidden_dim]
        
        rel_logits = self.relation_head(refined_repr) # [B, max_hops, num_relations]
        stop_logits = self.adaptive_stop_head(refined_repr).squeeze(-1) # [B, max_hops]
        
        return {
            'domain_logits': domain_logits,
            'confidence': q_confidence,
            'rel_logits': rel_logits,
            'stop_logits': stop_logits
        }

# ============================================================
#  Data & Training
# ============================================================

class UnifiedDataset(Dataset):
    def __init__(self, data_path, relation2id, domain2id, max_hops=4):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.samples = []
        for item in data:
            path = find_reasoning_path(item['sparql'])
            if path is None: continue
            
            # Domain from first relation path
            main_rel = path[0][1]
            domain = main_rel.split('.')[0] if '.' in main_rel else 'none'
            if domain not in domain2id: domain = 'none'
            
            # Relation IDs for each hop
            rel_ids = []
            valid = True
            for _, rel, _, _ in path:
                if rel in relation2id:
                    rel_ids.append(relation2id[rel])
                else:
                    valid = False; break
            
            if valid:
                # pad/truncate
                num_hops = len(rel_ids)
                if num_hops > max_hops: rel_ids = rel_ids[:max_hops]
                else: rel_ids = rel_ids + [0]*(max_hops - num_hops)
                
                self.samples.append({
                    'question': item['question'],
                    'domain': domain2id[domain],
                    'path': rel_ids,
                    'num_hops': min(num_hops, max_hops)
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_unified(batch, tokenizer):
    questions = [s['question'] for s in batch]
    domains = torch.tensor([s['domain'] for s in batch])
    paths = torch.tensor([s['path'] for s in batch])
    nums = torch.tensor([s['num_hops'] for s in batch])
    
    encoded = tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors='pt')
    return encoded, domains, paths, nums

def train_unified():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Maps
    rel2id = torch.load('data/processed_entity/relation2id.pt')
    dom2id = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id)
    num_dom = len(dom2id)
    
    # Setup Model
    model = UnifiedKGQAPlanner(num_dom, num_rel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Dataset
    train_ds = UnifiedDataset('data/cwq_train.json', rel2id, dom2id)
    dev_ds = UnifiedDataset('data/cwq_dev.json', rel2id, dom2id)
    collate = functools.partial(collate_unified, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=collate)
    
    # Loop
    epochs = 30
    best_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda')
    
    metrics_dir = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp6_unified.csv")
    with open(metrics_path, "w") as f:
        f.write("epoch,dev_loss\n")
    
    print("\nTraining Unified Adaptive-CHCP (Exp 6)...")
    for epoch in range(epochs):
        model.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for enc, doms, paths, nums in t_bar:
            enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)
            
            with torch.amp.autocast('cuda'):
                out = model(enc['input_ids'], enc['attention_mask'])
                
                # 1. Domain Loss
                loss_dom = F.cross_entropy(out['domain_logits'], doms)
                
                # 2. Relation Planning Loss
                # paths: [B, max_hops]
                loss_rel = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                
                # 3. Stop Loss
                # Binary target for each hop: 1 if hop is valid, 0 if it's padding
                # nums: [B]
                B, H = paths.size()
                stop_targets = torch.zeros(B, H).to(device)
                for b in range(B):
                    stop_targets[b, :nums[b]] = 1.0
                
                loss_stop = F.binary_cross_entropy_with_logits(out['stop_logits'], stop_targets)
                
                total_loss = loss_dom + loss_rel + loss_stop
            
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            t_bar.set_postfix(loss=total_loss.item())
            
        # Eval
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for enc, doms, paths, nums in dev_loader:
                enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)
                out = model(enc['input_ids'], enc['attention_mask'])
                loss = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                v_loss += loss.item()
        
        avg_v = v_loss / len(dev_loader)
        print(f"Epoch {epoch} | Dev Rel Loss: {avg_v:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_v:.4f}\n")
        
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), 'checkpoints/exp6_unified_best.pt')

if __name__ == "__main__":
    train_unified()
