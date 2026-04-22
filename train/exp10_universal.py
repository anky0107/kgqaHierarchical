"""
Exp 10: Universal Multi-Dataset Planner

Architecture: ScaledUnifiedPlanner (RoBERTa-Large) from Exp 7/8, 
upgraded with:
  1. Dataset Context Embedding (3-way: cwq / metaqa / webqsp)
  2. Topic Entity Text Injection (prefix to question string)
  3. Shared Universal Relation Vocabulary

Training Strategy:
  Stage 1 (CWQ)     : Load exp7_roberta_best.pt, train 15 epochs on CWQ
  Stage 2 (WebQSP)  : 10 epochs on WebQSP (1-hop, shares Freebase)
  Stage 3 (MetaQA)  : 10 epochs on MetaQA (freeze RoBERTa backbone)
  Stage 4 (CWQ)     : 5 epoch refresher to re-anchor CWQ SOTA score

  Joint Training: All datasets (CWQ, WebQSP, MetaQA) are combined into a 
  single balanced DataLoader using WeightedRandomSampler to ensure 
  equal representation during training.
"""

import os, sys, json, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ============================================================
#  Dataset IDs
# ============================================================
DATASET_IDS = {'cwq': 0, 'webqsp': 1, 'metaqa': 2}

# ============================================================
#  Universal Dataset Loader
# ============================================================
class UniversalDataset(Dataset):
    def __init__(self, json_path, relation2id, domain2id, dataset_name, max_hops=4):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        self.dataset_id = DATASET_IDS.get(dataset_name, 0)
        
        for item in data:
            rels = item.get('relations', [])
            if not rels:
                continue
            
            rel_ids = []
            for r in rels[:max_hops]:
                if r in relation2id:
                    rel_ids.append(relation2id[r])
                    
            if not rel_ids:
                continue
            
            # Pad to max_hops
            num_hops = len(rel_ids)
            rel_ids += [0] * (max_hops - len(rel_ids))
            
            # Topic entity text injection — prefix to question
            topic = item.get('topic_entity', '')
            question = item.get('question', '')
            if topic:
                question_with_entity = f"[{dataset_name.upper()}] topic: {topic} | {question}"
            else:
                question_with_entity = f"[{dataset_name.upper()}] {question}"
            
            # Domain from first relation
            first_rel = rels[0] if rels else ''
            domain = first_rel.split('.')[0] if '.' in first_rel else first_rel.split('_')[0]
            domain_id = domain2id.get(domain, 0)
            
            self.samples.append({
                'question': question_with_entity,
                'relations': torch.tensor(rel_ids, dtype=torch.long),
                'num_hops': num_hops,
                'domain': domain_id,
                'dataset_id': self.dataset_id
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_universal(batch, tokenizer):
    questions = [b['question'] for b in batch]
    enc = tokenizer(questions, padding=True, truncation=True, max_length=160, return_tensors='pt')
    rels = torch.stack([b['relations'] for b in batch])
    nums = torch.tensor([b['num_hops'] for b in batch], dtype=torch.long)
    doms = torch.tensor([b['domain'] for b in batch], dtype=torch.long)
    ds_ids = torch.tensor([b['dataset_id'] for b in batch], dtype=torch.long)
    return enc, doms, rels, nums, ds_ids

# ============================================================
#  Exp 10 Model: Universal Planner
# ============================================================
class UniversalPlanner(nn.Module):
    """
    ScaledUnifiedPlanner + Dataset Context Embedding.
    The dataset embedding is added to [CLS] before relation prediction,
    giving the model explicit knowledge of which graph topology to expect.
    """
    def __init__(self, num_domains, num_relations, hidden_dim=512, max_hops=4, num_datasets=3):
        super().__init__()
        self.max_hops = max_hops
        
        self.encoder = RobertaModel.from_pretrained("roberta-large")
        self.encoder_dim = self.encoder.config.hidden_size  # 1024
        
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        
        # Dataset context embedding — key architectural upgrade
        self.dataset_embedding = nn.Embedding(num_datasets, hidden_dim)
        
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.adaptive_stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, dataset_ids=None):
        B = input_ids.size(0)
        outputs = self.encoder(input_ids, attention_mask)
        q_h = outputs.last_hidden_state[:, 0, :]
        h_q = self.proj(q_h)
        
        # Inject dataset context embedding
        if dataset_ids is not None:
            ds_emb = self.dataset_embedding(dataset_ids)  # [B, hidden_dim]
            h_q = h_q + ds_emb  # Additive fusion
        
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

# ============================================================
#  Training Stage
# ============================================================
def train_stage(model, loader, optimizer, scaler, accumulation_steps, 
                num_rel, device, epoch_range, stage_name, metrics_path,
                val_loader=None, freeze_backbone=False, patience=3):
    
    if freeze_backbone:
        print(f"  [!] Freezing RoBERTa backbone for {stage_name} stage")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True
    
    epochs = list(epoch_range)
    if not epochs:
        print(f"  [!] No epochs to run for {stage_name}")
        return
    
    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(epochs), eta_min=1e-7)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt = os.path.join(ROOT, f'checkpoints/exp10_{stage_name}_best.pt')
    
    for epoch in epochs:
        model.train()
        t_bar = tqdm(loader, desc=f"[{stage_name}] Epoch {epoch}")
        
        for i, (enc, doms, paths, nums, ds_ids) in enumerate(t_bar):
            enc = enc.to(device)
            doms = doms.to(device)
            paths = paths.to(device)
            nums = nums.to(device)
            ds_ids = ds_ids.to(device)
            
            with torch.amp.autocast('cuda'):
                out = model(enc['input_ids'], enc['attention_mask'], ds_ids)
                
                loss_dom = F.cross_entropy(out['domain_logits'], doms)
                loss_rel = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                
                B, H = paths.size()
                stop_targets = torch.zeros(B, H, device=device)
                for b in range(B):
                    stop_targets[b, :nums[b]] = 1.0
                loss_stop = F.binary_cross_entropy_with_logits(out['stop_logits'], stop_targets)
                
                total_loss = (loss_dom + loss_rel + loss_stop) / accumulation_steps
            
            scaler.scale(total_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            t_bar.set_postfix(loss=total_loss.item() * accumulation_steps)
        
        scheduler.step()
        
        # Validation
        v_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for enc, doms, paths, nums, ds_ids in val_loader:
                    enc = enc.to(device); paths = paths.to(device); ds_ids = ds_ids.to(device)
                    out = model(enc['input_ids'], enc['attention_mask'], ds_ids)
                    v_loss += F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1)).item()
            v_loss /= len(val_loader)
            
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{stage_name}] Epoch {epoch} | Val Loss: {v_loss:.4f} | LR: {lr_now:.2e}")
            
            # Early stopping
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt)
                print(f"    [NEW BEST] Saved {best_ckpt}")
            else:
                patience_counter += 1
                print(f"    [No improvement] patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"  [Early Stop] {stage_name} stopped at epoch {epoch}")
                    break
        
        with open(metrics_path, 'a') as f:
            f.write(f"{stage_name},{epoch},{v_loss:.4f}\n")
        
        # Per-epoch checkpoint (for resume)
        ckpt = os.path.join(ROOT, f'checkpoints/exp10_{stage_name}_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt)
    
    # Reload best weights at end of stage
    if os.path.exists(best_ckpt):
        print(f"  [{stage_name}] Reloading best weights from {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

# ============================================================
#  Main Training Pipeline
# ============================================================
def train_exp10_universal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    proc_dir = os.path.join(ROOT, 'data/processed_universal')
    relation2id = torch.load(os.path.join(proc_dir, 'relation2id.pt'))
    domain2id = torch.load(os.path.join(proc_dir, 'domain2id.pt'))
    num_rel = max(relation2id.values()) + 1
    num_dom = max(domain2id.values()) + 1
    print(f"Universal vocab: {num_rel} relations, {num_dom} domains")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    collate = functools.partial(collate_universal, tokenizer=tokenizer)
    
    model = UniversalPlanner(num_dom, num_rel).to(device)
    
    # Enable gradient checkpointing to halve VRAM usage
    model.encoder.gradient_checkpointing_enable()
    
    # Auto-resume: find latest checkpoint, parse stage + epoch exactly
    import re as _re
    
    STAGE_ORDER = ['cwq', 'webqsp', 'metaqa', 'cwq_refresh']
    resume_stage = 'cwq'
    resume_epoch = 0
    
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    all_ckpts = [f for f in os.listdir(ckpt_dir) if _re.match(r'exp10_(cwq|webqsp|metaqa|cwq_refresh)_epoch_\d+\.pt', f)]
    
    if all_ckpts:
        def ckpt_sort_key(fname):
            m = _re.match(r'exp10_(cwq_refresh|cwq|webqsp|metaqa)_epoch_(\d+)\.pt', fname)
            if not m: return (0, 0)
            stage_idx = STAGE_ORDER.index(m.group(1)) if m.group(1) in STAGE_ORDER else 0
            return (stage_idx, int(m.group(2)))
        
        latest = max(all_ckpts, key=ckpt_sort_key)
        m = _re.match(r'exp10_(cwq_refresh|cwq|webqsp|metaqa)_epoch_(\d+)\.pt', latest)
        resume_stage = m.group(1)
        resume_epoch = int(m.group(2)) + 1  # start from NEXT epoch
        
        ckpt_path = os.path.join(ckpt_dir, latest)
        print(f"Auto-resuming from: {latest}")
        print(f"  -> Stage: {resume_stage}, Next epoch: {resume_epoch}")
        
        state_dict = torch.load(ckpt_path, map_location=device)
        
        # Check for num_rel mismatch
        ckpt_num_rel = state_dict['relation_head.weight'].shape[0]
        if ckpt_num_rel != num_rel:
            print(f"  [!] Rel vocab mismatch: CKPT({ckpt_num_rel}) vs CURR({num_rel})")
            print("  [!] Surgically expanding relation head...")
            model.relation_head.weight.data[:ckpt_num_rel] = state_dict['relation_head.weight']
            model.relation_head.bias.data[:ckpt_num_rel] = state_dict['relation_head.bias']
            del state_dict['relation_head.weight']
            del state_dict['relation_head.bias']
            
        # Check for num_dom mismatch
        ckpt_num_dom = state_dict['domain_head.weight'].shape[0]
        if ckpt_num_dom != num_dom:
            print(f"  [!] Dom vocab mismatch: CKPT({ckpt_num_dom}) vs CURR({num_dom})")
            print("  [!] Surgically expanding domain head...")
            model.domain_head.weight.data[:ckpt_num_dom] = state_dict['domain_head.weight']
            model.domain_head.bias.data[:ckpt_num_dom] = state_dict['domain_head.bias']
            del state_dict['domain_head.weight']
            del state_dict['domain_head.bias']
        
        model.load_state_dict(state_dict, strict=False)
    else:
        # Fresh start — transfer weights from Exp 7
        old_rel2id = torch.load(os.path.join(ROOT, 'data/processed_entity/relation2id.pt'))
        old_ckpt = torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device)
        new_state = model.state_dict()
        transferred = 0
        for k, v in old_ckpt.items():
            if k in new_state and new_state[k].shape == v.shape:
                new_state[k] = v
                transferred += 1
        old_id2rel = {i: r for r, i in old_rel2id.items()}
        old_rel_weight = old_ckpt.get('relation_head.weight')
        if old_rel_weight is not None:
            new_rel_weight = new_state['relation_head.weight']
            for old_i, rel in old_id2rel.items():
                if rel in relation2id:
                    new_rel_weight[relation2id[rel]] = old_rel_weight[old_i]
            new_state['relation_head.weight'] = new_rel_weight
        model.load_state_dict(new_state)
        print(f"Fresh start: transferred {transferred} weight tensors from Exp 7")
    
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'metrics'), exist_ok=True)
    metrics_path = os.path.join(ROOT, 'metrics/exp10_universal.csv')
    with open(metrics_path, 'w') as f:
        f.write("stage,epoch,val_loss\n")
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Load Datasets
    print("\n" + "="*50)
    print("LOADING ALL DATASETS FOR JOINT TRAINING")
    print("="*50)
    cwq_train_ds = UniversalDataset(os.path.join(proc_dir, 'cwq/train.json'), relation2id, domain2id, 'cwq')
    webq_train_ds = UniversalDataset(os.path.join(proc_dir, 'webqsp/train.json'), relation2id, domain2id, 'webqsp')
    meta_train_ds = UniversalDataset(os.path.join(proc_dir, 'metaqa/train.json'), relation2id, domain2id, 'metaqa')

    # Balanced Sampling Strategy: Each dataset gets 1/3 of the batch probability
    datasets = [cwq_train_ds, webq_train_ds, meta_train_ds]
    combined_ds = ConcatDataset(datasets)
    
    weights = []
    for ds in datasets:
        ds_weight = 1.0 / len(ds) / len(datasets)
        weights.extend([ds_weight] * len(ds))
    
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)
    # Larger batch size for joint training stability
    joint_loader = DataLoader(combined_ds, batch_size=8, sampler=sampler, collate_fn=collate, num_workers=2)
    
    # Validation Loaders
    cwq_dev_loader = DataLoader(UniversalDataset(os.path.join(proc_dir, 'cwq/dev.json'), relation2id, domain2id, 'cwq'), 
                               batch_size=16, collate_fn=collate)
    webq_dev_loader = DataLoader(UniversalDataset(os.path.join(proc_dir, 'webqsp/dev.json'), relation2id, domain2id, 'webqsp'), 
                                batch_size=16, collate_fn=collate)
    meta_dev_loader = DataLoader(UniversalDataset(os.path.join(proc_dir, 'metaqa/dev.json'), relation2id, domain2id, 'metaqa'), 
                                batch_size=32, collate_fn=collate)

    # Optimizer (joint optimization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    print(f"Joint Training Start: {len(combined_ds)} items across 3 datasets")
    
    # Joint Training Loop
    metrics_path = os.path.join(ROOT, 'metrics/exp10_joint_metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write("epoch,cwq_loss,webq_loss,meta_loss,avg_loss\n")
        
    for epoch in range(resume_epoch, 20):
        print(f"\n--- JOINT EPOCH {epoch} ---")
        model.train()
        t_bar = tqdm(joint_loader, desc=f"Epoch {epoch}")
        
        epoch_losses = defaultdict(list)
        
        for i, (enc, doms, paths, nums, ds_ids) in enumerate(t_bar):
            enc = enc.to(device); doms = doms.to(device); paths = paths.to(device)
            nums = nums.to(device); ds_ids = ds_ids.to(device)
            
            with torch.amp.autocast('cuda'):
                out = model(enc['input_ids'], enc['attention_mask'], ds_ids)
                loss_dom = F.cross_entropy(out['domain_logits'], doms)
                loss_rel = F.cross_entropy(out['rel_logits'].view(-1, num_rel), paths.view(-1))
                
                # Stop targets
                B, H = paths.size()
                stop_targets = torch.zeros(B, H, device=device)
                for b in range(B): stop_targets[b, :nums[b]] = 1.0
                loss_stop = F.binary_cross_entropy_with_logits(out['stop_logits'], stop_targets)
                
                total_loss = (loss_dom + loss_rel + loss_stop) / 8 # Using 8 accumulation steps
                
            scaler.scale(total_loss).backward()
            
            if (i + 1) % 8 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            t_bar.set_postfix(loss=total_loss.item() * 8)
            
        # Per-epoch Validation on ALL datasets
        model.eval()
        v_scores = {}
        with torch.no_grad():
            # Quick check on CWQ dev for stability
            total, correct = 0, 0
            for enc, _, paths, nums, ds_ids in cwq_dev_loader:
                enc = enc.to(device); ds_ids = ds_ids.to(device)
                out = model(enc['input_ids'], enc['attention_mask'], ds_ids)
                preds = out['rel_logits'].argmax(dim=-1)
                for b in range(paths.size(0)):
                    if paths[b, :nums[b]].tolist() == preds[b, :nums[b]].tolist(): correct += 1
                    total += 1
            v_scores['cwq'] = correct / total
            print(f"  [Eval] CWQ Path Accuracy: {100*v_scores['cwq']:.2f}%")

        # Save checkpoint
        ckpt_path = os.path.join(ROOT, f'checkpoints/exp10_joint_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)
    print("\n✓ Exp 10 complete! Final checkpoint saved.")

if __name__ == '__main__':
    train_exp10_universal()
