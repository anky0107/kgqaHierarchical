"""
Exp 7: Scaling to RoBERTa-Large

Pushing the Unified Planner (Exp 6) architecture to its capacity limit using RoBERTa-Large.
"""
import os, sys, json, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp6_unified import UnifiedDataset, collate_unified

# ============================================================
#  Scaled Model Architecture (RoBERTa-Large)
# ============================================================

class ScaledUnifiedPlanner(nn.Module):
    def __init__(self, num_domains, num_relations, hidden_dim=512, max_hops=4):
        super().__init__()
        self.max_hops = max_hops
        
        # Scaling to RoBERTa-Large
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.encoder = RobertaModel.from_pretrained("roberta-large")
        self.encoder_dim = self.encoder.config.hidden_size # 1024
        
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4) # Deeper transformer
        
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.adaptive_stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)
        outputs = self.encoder(input_ids, attention_mask)
        q_h = outputs.last_hidden_state[:, 0, :] # CLS token
        h_q = self.proj(q_h)
        
        domain_logits = self.domain_head(h_q)
        q_confidence = torch.sigmoid(self.confidence_head(h_q))
        
        init_repr = h_q.unsqueeze(1) + self.hop_embeddings.unsqueeze(0)
        refined_repr = self.transformer(init_repr)
        
        rel_logits = self.relation_head(refined_repr)
        stop_logits = self.adaptive_stop_head(refined_repr).squeeze(-1)
        
        return {
            'domain_logits': domain_logits,
            'confidence': q_confidence,
            'rel_logits': rel_logits,
            'stop_logits': stop_logits
        }

def train_roberta_scaled():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Maps
    rel2id = torch.load('data/processed_entity/relation2id.pt')
    dom2id = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id)
    num_dom = len(dom2id)
    
    # Model (smaller hidden_dim to fit 8GB VRAM)
    model = ScaledUnifiedPlanner(num_dom, num_rel, hidden_dim=512).to(device)
    
    start_epoch = 0
    roberta_ckpts = [f for f in os.listdir(os.path.join(ROOT, 'checkpoints')) if f.startswith('exp7_roberta_epoch_') and f.endswith('.pt')]
    if roberta_ckpts:
        latest_ckpt = max(roberta_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0]) + 1
        ckpt_path = os.path.join(ROOT, 'checkpoints', latest_ckpt)
        print(f"Resuming Exp 7 from {ckpt_path} (Starting at Epoch {start_epoch})")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    # Dataset
    train_ds = UnifiedDataset('data/cwq_train.json', rel2id, dom2id)
    dev_ds = UnifiedDataset('data/cwq_dev.json', rel2id, dom2id)
    collate = functools.partial(collate_unified, tokenizer=tokenizer)
    
    # Batch size 4 + Gradient Accumulation to fit in 8GB VRAM
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=8, collate_fn=collate)
    
    epochs = 30
    best_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = 4 

    metrics_dir = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp7_roberta.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,dev_loss\n")

    print(f"\nStarting Final Training: Escalate to RoBERTa-Large (Exp 7)...")
    for epoch in range(start_epoch, epochs):
        model.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for i, (enc, doms, paths, nums) in enumerate(t_bar):
            enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)
            
            with torch.amp.autocast('cuda'):
                out = model(enc['input_ids'], enc['attention_mask'])
                loss_dom = F.cross_entropy(out['domain_logits'], doms)
                loss_rel = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                
                B, H = paths.size()
                stop_targets = torch.zeros(B, H).to(device)
                for b in range(B):
                    stop_targets[b, :nums[b]] = 1.0
                loss_stop = F.binary_cross_entropy_with_logits(out['stop_logits'], stop_targets)
                
                total_loss = (loss_dom + loss_rel + loss_stop) / accumulation_steps
            
            scaler.scale(total_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            t_bar.set_postfix(loss=total_loss.item() * accumulation_steps)
            
            t_bar.set_postfix(loss=total_loss.item() * accumulation_steps)
            
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
            torch.save(model.state_dict(), f'checkpoints/exp7_roberta_best.pt')
        
        # Keep tracking by epoch since it's the last experiment
        torch.save(model.state_dict(), f'checkpoints/exp7_roberta_epoch_{epoch}.pt')

if __name__ == "__main__":
    train_roberta_scaled()
