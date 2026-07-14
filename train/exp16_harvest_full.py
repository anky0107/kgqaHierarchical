import os, sys, json, torch, lmdb, pickle
from tqdm import tqdm
from transformers import RobertaTokenizer, AutoTokenizer, AutoModel
import torch.nn.functional as F

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from inference_pipeline.model import ScaledUnifiedPlanner
from utils.sparql_parser import find_reasoning_path

class OptimizedCDSDataCollector:
    def __init__(self, exp15_ckpt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Collector] Initializing Optimized Harvest on {self.device}")
        
        # 1. Load Exp15 Agent
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        
        base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.agent = STRLAgent(base).to(self.device)
        self.agent.load_state_dict(torch.load(exp15_ckpt, map_location=self.device))
        self.agent.eval()
        
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # 2. Optimized Stage 1 Bi-Encoder (Pre-computed Lookup)
        self.s1_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.s1_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.s1_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp16_s1_bi.pt'), map_location=self.device))
        self.s1_model.eval()
        
        print("[Collector] Loading Pre-computed Embeddings...")
        data = torch.load(os.path.join(ROOT, 'data/exp16_entity_embs.pt'), map_location='cpu')
        self.all_mids = data['mids']
        self.mid2idx = {mid: i for i, mid in enumerate(self.all_mids)}
        self.all_embs = data['embs'].to(self.device)
        del data
        
        # 3. KG & Mappings
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

    def kg_lookup(self, entities, rels):
        next_entities = set()
        with self.env.begin() as txn:
            for ent in entities:
                for rel in rels:
                    f_data = txn.get(f"f:{ent}".encode('utf-8'))
                    if f_data:
                        for r, tgt in pickle.loads(f_data):
                            if r == rel: next_entities.add(tgt)
                    b_data = txn.get(f"b:{ent}".encode('utf-8'))
                    if b_data:
                        for r, src in pickle.loads(b_data):
                            if r == rel: next_entities.add(src)
        return next_entities

    @torch.no_grad()
    def collect(self, input_file, output_file, limit=None):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if limit: data = data[:limit]
        
        results = []
        for item in tqdm(data, desc=f"Harvesting {os.path.basename(input_file)}"):
            try:
                path = find_reasoning_path(item['sparql'])
                if not path: continue
                
                q = item['question']
                te = path[0][0].replace("ns:", "")
                gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
                
                # Traversal
                inputs = self.tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(self.device)
                fwd = self.agent(inputs['input_ids'], inputs['attention_mask'])
                
                current = {te}
                execution_log = []
                for h in range(4):
                    action = torch.argmax(fwd['action_logits'][0, h]).item()
                    if action == 3: break
                    hop_repr = fwd['hop_reprs'][0, h]
                    sims = torch.mv(self.rel_emb_bank.all(), hop_repr)
                    k = {0:5, 1:10, 2:50}.get(action, 5)
                    top_k = torch.topk(sims, k).indices.tolist()
                    rels = [self.id2rel[rid] for rid in top_k]
                    execution_log.append(rels[0]) # save top rel name
                    current = self.kg_lookup(current, rels)
                    if not current: break
                
                # Optimized Pruning for Training Data
                # (We keep more candidates for the ranker to learn from)
                q_enc = self.s1_tok(q, return_tensors='pt', padding=True, truncation=True).to(self.device)
                q_emb = self.s1_model(**q_enc).last_hidden_state[:, 0, :]
                
                candidates = []
                for mid in current:
                    name = self.mid2name.get(mid, "Unknown")
                    is_gold = mid in gold
                    
                    # For training, we want to know the S1 score
                    if mid in self.mid2idx:
                        e_emb = self.all_embs[self.mid2idx[mid]]
                        s1_score = F.cosine_similarity(q_emb, e_emb.unsqueeze(0)).item()
                    else:
                        s1_score = 0.0
                        
                    candidates.append({
                        'mid': mid, 'name': name, 'is_gold': is_gold, 's1_score': s1_score
                    })
                
                results.append({
                    'question': q,
                    'path': " -> ".join(execution_log),
                    'candidates': candidates
                })
            except: pass
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"[Collector] Saved {len(results)} samples.")

if __name__ == "__main__":
    collector = OptimizedCDSDataCollector(os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_19.pt'))
    # FULL HARVEST
    collector.collect(os.path.join(ROOT, 'data/cwq_train.json'), os.path.join(ROOT, 'data/exp16_cds_train_full.json'))
