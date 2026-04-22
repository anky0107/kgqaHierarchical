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
#  Universal Dataset Loader (No Tags)
# ============================================================
class UntaggedUniversalDataset(Dataset):
    def __init__(self, json_path, relation2id, max_hops=4):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            rels = item.get('relations', [])
            if not rels: continue
            
            rel_ids = [relation2id[r] for r in rels[:max_hops] if r in relation2id]
            if not rel_ids: continue
            
            num_hops = len(rel_ids)
            rel_ids += [0] * (max_hops - len(rel_ids))
            
            # Remove dataset tags entirely, we want true semantic understanding
            question = item.get('question', '')
            topic = item.get('topic_entity', '')
            question_with_topic = f"topic: {topic} | {question}" if topic else question
            
            self.samples.append({
                'question': question_with_topic,
                'relations': torch.tensor(rel_ids, dtype=torch.long),
                'num_hops': num_hops
            })
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_untagged(batch, tokenizer):
    questions = [b['question'] for b in batch]
    enc = tokenizer(questions, padding=True, truncation=True, max_length=160, return_tensors='pt')
    rels = torch.stack([b['relations'] for b in batch])
    nums = torch.tensor([b['num_hops'] for b in batch], dtype=torch.long)
    return enc, rels, nums

# ============================================================
#  Exp 11 Model: Vector Projection Planner (UltraQuery-style)
# ============================================================
class VectorProjectionPlanner(nn.Module):
    def __init__(self, relation_embeddings, hidden_dim=512, max_hops=4):
        super().__init__()
        self.max_hops = max_hops
        
        self.encoder = RobertaModel.from_pretrained("roberta-large")
        self.encoder_dim = self.encoder.config.hidden_size  # 1024
        
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Instead of a fixed classification head, we project to the relation embedding space
        embedding_dim = relation_embeddings.size(1)
        self.vector_head = nn.Linear(hidden_dim, embedding_dim)
        
        # Frozen relation embeddings for cosine similarity
        self.register_buffer('relation_embeddings', relation_embeddings)

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)
        outputs = self.encoder(input_ids, attention_mask)
        q_h = outputs.last_hidden_state[:, 0, :]
        h_q = self.proj(q_h)
        
        q_confidence = torch.sigmoid(self.confidence_head(h_q))
        
        init_repr = h_q.unsqueeze(1) + self.hop_embeddings.unsqueeze(0)
        refined_repr = self.transformer(init_repr)
        
        # [B, H, embedding_dim]
        predicted_vectors = self.vector_head(refined_repr)
        
        # Normalize for cosine similarity
        predicted_vectors = F.normalize(predicted_vectors, p=2, dim=-1) # [B, H, E]
        target_embeddings = F.normalize(self.relation_embeddings, p=2, dim=-1) # [NumRel, E]
        
        # Dot product gives Cosine Similarity logits [B, H, NumRel]
        similarity_logits = torch.matmul(predicted_vectors, target_embeddings.T)
        
        # Scale for softmax sharpness (temperature)
        similarity_logits = similarity_logits * 20.0
        
        return similarity_logits

# Pre-compute relation semantics mapping
def build_relation_embeddings(relation2id, model_name="roberta-base"):
    print("Pre-computing Relation Vocabulary Embeddings...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    encoder = RobertaModel.from_pretrained(model_name)
    encoder.eval()
    
    num_rel = max(relation2id.values()) + 1
    # Sort relations by ID
    id2rel = {v: k for k, v in relation2id.items()}
    relations = [id2rel.get(i, "unknown") for i in range(num_rel)]
    
    # Textualize relations: "film.film.directed_by" -> "film film directed by"
    clean_relations = [r.replace('.', ' ').replace('_', ' ') for r in relations]
    
    encoded = tokenizer(clean_relations, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = encoder(**encoded)
        # Use CLS token as relation representation
        rel_embs = out.last_hidden_state[:, 0, :]
        
    print(f"  -> Generated {rel_embs.size()} embeddings matrix.")
    return rel_embs

def train_exp11_vector_projection():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join(ROOT, 'data/processed_universal')
    relation2id = torch.load(os.path.join(proc_dir, 'relation2id.pt'))
    num_rel = max(relation2id.values()) + 1
    
    # 1. Build relation vectors
    rel_emb_path = os.path.join(proc_dir, 'relation_embeddings.pt')
    if os.path.exists(rel_emb_path):
        rel_embeddings = torch.load(rel_emb_path)
    else:
        rel_embeddings = build_relation_embeddings(relation2id)
        torch.save(rel_embeddings, rel_emb_path)
    rel_embeddings = rel_embeddings.to(device)

    # 2. Datasets & Loaders
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    collate = functools.partial(collate_untagged, tokenizer=tokenizer)
    
    cwq_train_ds = UntaggedUniversalDataset(os.path.join(proc_dir, 'cwq/train.json'), relation2id)
    webq_train_ds = UntaggedUniversalDataset(os.path.join(proc_dir, 'webqsp/train.json'), relation2id)
    meta_train_ds = UntaggedUniversalDataset(os.path.join(proc_dir, 'metaqa/train.json'), relation2id)

    datasets = [cwq_train_ds, webq_train_ds, meta_train_ds]
    combined_ds = ConcatDataset(datasets)
    
    weights = []
    for ds in datasets:
        ds_weight = 1.0 / len(ds) / len(datasets)
        weights.extend([ds_weight] * len(ds))
    
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)
    joint_loader = DataLoader(combined_ds, batch_size=8, sampler=sampler, collate_fn=collate, num_workers=2)

    # 3. Model & Optimizer
    model = VectorProjectionPlanner(rel_embeddings).to(device)
    model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scaler = torch.amp.GradScaler('cuda')
    train.set = dataset.load("data/processed_universal/cwq/train.json")
    
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    
    # 4. Train Loop
    print("\nStarting Exp 11: Vector Projection (Schema-Agnostic)")
    for epoch in range(5):
        model.train()
        t_bar = tqdm(joint_loader, desc=f"Ep {epoch} Vector Proj")
        
        for i, (enc, paths, nums) in enumerate(t_bar):
            enc = enc.to(device); paths = paths.to(device); nums = nums.to(device)
            
            with torch.amp.autocast('cuda'):
                # Forward pass gives direct similarity logits
                rel_logits = model(enc['input_ids'], enc['attention_mask'])
                
                loss_rel = F.cross_entropy(rel_logits.view(-1, num_rel), paths.view(-1))
                loss = loss_rel / 8 # Acc steps
                
            scaler.scale(loss).backward()
            if (i + 1) % 8 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            t_bar.set_postfix(loss=loss.item() * 8)
            
        torch.save(model.state_dict(), os.path.join(ROOT, f'checkpoints/exp11_vector_epoch_{epoch}.pt'))

if __name__ == '__main__':
    train_exp11_vector_projection()
