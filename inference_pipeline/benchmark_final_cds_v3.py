import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F, lmdb, pickle
from transformers import RobertaTokenizer, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from utils.sparql_parser import find_reasoning_path

class PathAwareRanker(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        q_emb = self.encoder(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p_emb = self.encoder(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e_emb = self.encoder(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q_emb, p_emb, e_emb], dim=-1)).squeeze(-1)

class CascadingDustSeparatorV3:
    """
    CDS v3: Path-Aware Joint Ranking.
    Final Stage 3 Cross-Encoder now sees both Question and Reasoning Path.
    """
    def __init__(self, device):
        self.device = device
        print("[CDS v3] Initializing Path-Aware Cascading Stack...")
        
        # S1: Bi-Encoder (MiniLM-L6)
        self.s1_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.s1_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        s1_path = os.path.join(ROOT, 'checkpoints/exp16v2_s1_bi.pt') # fallback to v2 if v3 not trained
        self.s1_model.load_state_dict(torch.load(s1_path, map_location=device))
        self.s1_model.eval()
        
        # Load Pre-computed Embeddings
        print("[CDS v3] Loading 10.8GB Embedding Bank to RAM...")
        data = torch.load(os.path.join(ROOT, 'data/exp16_entity_embs.pt'), map_location='cpu')
        self.all_mids = data['mids']
        self.mid2idx = {mid: i for i, mid in enumerate(self.all_mids)}
        self.all_embs = data['embs']
        del data
        
        # S2: Path-Aware Fusion (mpnet-base-v2)
        self.s2_model = PathAwareRanker().to(device)
        s2_path = os.path.join(ROOT, 'checkpoints/exp16v2_s2_path.pt')
        self.s2_model.load_state_dict(torch.load(s2_path, map_location=device))
        self.s2_model.eval()
        
        # S3: PATH-AWARE Cross-Encoder (BGE-reranker-base)
        print("[CDS v3] Loading Path-Aware Stage 3 (BGE-reranker)...")
        self.s3_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.s3_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
        s3_path = os.path.join(ROOT, 'checkpoints/exp16v3_s3_cross.pt')
        if os.path.exists(s3_path):
            self.s3_model.load_state_dict(torch.load(s3_path, map_location=device))
        else:
            print("[WARNING] v3 checkpoint not found, using v2 cross-encoder (non-path-aware fallback)")
            self.s3_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp16v2_s3_cross.pt'), map_location=device))
        self.s3_model.eval()
        
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

    @torch.no_grad()
    def separate_dust(self, question, path_str, candidate_mids):
        if not candidate_mids: return None
        
        # --- Stage 1: Pruning ---
        q_enc = self.s1_tok(question, return_tensors='pt', padding=True, truncation=True).to(self.device)
        q_emb = self.s1_model(**q_enc).last_hidden_state[:, 0, :].cpu()
        
        found_mids = [m for m in candidate_mids if m in self.mid2idx]
        if not found_mids: return list(candidate_mids)[0]
        
        e_embs = self.all_embs[[self.mid2idx[m] for m in found_mids]]
        sims = F.cosine_similarity(q_emb, e_embs)
        
        top_k1 = min(100, len(found_mids))
        top_idx1 = torch.topk(sims, top_k1).indices.tolist()
        mids1 = [found_mids[i] for i in top_idx1]
        
        # --- Stage 2: Path-Sieve ---
        names1 = [self.mid2name.get(m, "Unknown") for m in mids1]
        q_enc = self.s1_tok([question]*len(mids1), return_tensors='pt', padding=True, truncation=True).to(self.device)
        p_enc = self.s1_tok([path_str]*len(mids1), return_tensors='pt', padding=True, truncation=True).to(self.device)
        e_enc = self.s1_tok(names1, return_tensors='pt', padding=True, truncation=True).to(self.device)
        
        scores2 = self.s2_model(q_enc['input_ids'], q_enc['attention_mask'],
                                p_enc['input_ids'], p_enc['attention_mask'],
                                e_enc['input_ids'], e_enc['attention_mask'])
        
        top_k2 = min(20, len(mids1))
        top_idx2 = torch.topk(scores2, top_k2).indices.tolist()
        mids2 = [mids1[i] for i in top_idx2]
        names2 = [names1[i] for i in top_idx2]
        
        # --- Stage 3: PATH-AWARE Judge ---
        # V3 Change: Segment 1 = Question + Path
        q_with_path = f"{question} [PATH] {path_str}"
        qs3 = [q_with_path] * len(mids2)
        enc3 = self.s3_tok(qs3, names2, return_tensors='pt', padding=True, truncation=True, max_length=192).to(self.device)
        logits3 = self.s3_model(**enc3).logits.squeeze(-1)
        final_idx = torch.argmax(logits3).item()
        
        return mids2[final_idx]

class MasterEvaluator:
    def __init__(self, model_type='exp15', ckpt_path=None, env=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Master] Evaluator ({model_type}) on {self.device}")
        self.model_type = model_type
        self.env = env
        
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Load Model
        if model_type == 'exp15':
            base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
            self.agent = STRLAgent(base).to(self.device)
        elif model_type == 'exp9':
            base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
            self.agent = RLConstraintAgent(base).to(self.device)
        elif model_type == 'exp7':
            self.agent = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.agent.eval()
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
        
        self.cds = CascadingDustSeparatorV3(self.device)
        self.env = env # Use shared env

    def kg_lookup(self, entities, rels):
        next_entities = set()
        rels_set = set(rels)
        with self.env.begin() as txn:
            for ent in entities:
                f_data = txn.get(f"f:{ent}".encode())
                if f_data:
                    for r, tgt in pickle.loads(f_data):
                        if r in rels_set: next_entities.add(tgt)
                b_data = txn.get(f"b:{ent}".encode())
                if b_data:
                    for r, src in pickle.loads(b_data):
                        if r in rels_set: next_entities.add(src)
        return next_entities

    def evaluate(self, samples, model_name="Model"):
        stats = {'hit1': 0, 'hit_n': 0, 'ents': 0, 'count': 0}
        for s in tqdm(samples, desc=f"Evaluating {model_name} (CDS v3)"):
            inputs = self.tokenizer(s['q'], return_tensors="pt", padding=True, truncation=True).to(self.device)
            current = {s['te']}
            path_log = []
            
            # Agent Reasoning
            with torch.no_grad():
                fwd = self.agent(inputs['input_ids'], inputs['attention_mask'])
                
                # Normalize outputs across different agent types
                if self.model_type == 'exp15':
                    action_logits = fwd['action_logits']
                    hop_reprs = fwd['hop_reprs']
                    rel_logits = None # Use hop_reprs for semantic matching
                elif self.model_type == 'exp9':
                    # RLConstraintAgent returns: action_logits, state_values, rel_logits, domain_logits
                    action_logits, _, rel_logits, _ = fwd
                    hop_reprs = None
                elif self.model_type == 'exp7':
                    # ScaledUnifiedPlanner returns: dict with rel_logits, stop_logits, dom_logits
                    action_logits = fwd['stop_logits']
                    rel_logits = fwd['rel_logits']
                    hop_reprs = None
                
                for h in range(4):
                    action = torch.argmax(action_logits[0, h]).item()
                    if action == 3: break
                    
                    if self.model_type == 'exp15':
                        sims = torch.mv(self.rel_emb_bank.all(), hop_reprs[0, h])
                    else:
                        sims = rel_logits[0, h]
                    
                    # k selection: Exp 9 uses actions 0-2 for breadth; Exp 7/15 use action 0-2 as "Continue"
                    k = {0:5, 1:10, 2:50}.get(action, 5)
                    top_k = torch.topk(sims, k).indices.tolist()
                    rel = self.id2rel[top_k[0]]
                    path_log.append(rel)
                    current = self.kg_lookup(current, [self.id2rel[rid] for rid in top_k])
                    if not current: break
            
            path_str = " -> ".join(path_log)
            stats['count'] += 1; stats['ents'] += len(current)
            if any(mid in s['gold'] for mid in current):
                stats['hit_n'] += 1
                pred = self.cds.separate_dust(s['q'], path_str, current)
                if pred in s['gold']: stats['hit1'] += 1
        
        print(f"\n[{model_name} + CDS v3] Result: Hit@1={ (stats['hit1']/stats['count'])*100:.2f}%, Recall={ (stats['hit_n']/stats['count'])*100:.2f}%")
        return stats

def main():
    with open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8') as f:
        data = json.load(f); samples = []
        for item in tqdm(data, desc="Preprocessing Dev Set"):
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            samples.append({'q': item['question'], 'te': path[0][0].replace("ns:", ""), 'gold': set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))})
    
    models = [
        ('exp9', os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt')),
        ('exp7', os.path.join(ROOT, 'checkpoints/exp7_roberta_epoch_19.pt'))
    ]
    
    lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    for m_type, m_ckpt in models:
        if not os.path.exists(m_ckpt): continue
        evaluator = MasterEvaluator(m_type, m_ckpt, env=env)
        evaluator.evaluate(samples, model_name=m_type.upper())

if __name__ == "__main__":
    main()
