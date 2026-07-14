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
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fuse = nn.Linear(self.encoder.config.hidden_size * 3, 1)
    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        q_emb = self.encoder(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p_emb = self.encoder(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e_emb = self.encoder(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q_emb, p_emb, e_emb], dim=-1)).squeeze(-1)

class CascadingDustSeparator:
    def __init__(self, device):
        self.device = device
        print("[CDS] Loading Triple-Stage Stack (CPU-Optimized for VRAM)...")
        
        # S1: Bi-Encoder (Weights for encoding question)
        self.s1_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.s1_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.s1_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp16_s1_bi.pt'), map_location=device))
        self.s1_model.eval()
        
        # Load Pre-computed Embeddings (KEEP ON CPU to save VRAM)
        print("[CDS] Loading Pre-computed Entity Embeddings to RAM...")
        data = torch.load(os.path.join(ROOT, 'data/exp16_entity_embs.pt'), map_location='cpu')
        self.all_mids = data['mids']
        self.mid2idx = {mid: i for i, mid in enumerate(self.all_mids)}
        self.all_embs = data['embs'] # Keep on CPU
        del data
        
        # S2: Path-Aware
        self.s2_model = PathAwareRanker().to(device)
        self.s2_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp16_cds_epoch_5.pt'), map_location=device))
        self.s2_model.eval()
        
        # S3: Cross-Encoder
        self.s3_tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.s3_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1).to(device)
        self.s3_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp16_s3_cross.pt'), map_location=device))
        self.s3_model.eval()
        
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

    @torch.no_grad()
    def separate_dust(self, question, path_str, candidate_mids):
        if not candidate_mids: return None
        
        # Stage 1: Bi-Encoder Pruning (CPU Cosine Similarity)
        q_enc = self.s1_tok(question, return_tensors='pt', padding=True, truncation=True).to(self.device)
        q_emb = self.s1_model(**q_enc).last_hidden_state[:, 0, :].cpu() # Move to CPU for similarity
        
        cand_idx = []
        found_candidates = []
        for mid in candidate_mids:
            if mid in self.mid2idx:
                cand_idx.append(self.mid2idx[mid])
                found_candidates.append(mid)
        
        if not cand_idx: return list(candidate_mids)[0]
            
        e_embs = self.all_embs[cand_idx]
        sims = F.cosine_similarity(q_emb, e_embs)
        
        top_k_val = min(100, len(found_candidates))
        top_idx_in_cand = torch.topk(sims, top_k_val).indices.tolist()
        candidates_mids = [found_candidates[i] for i in top_idx_in_cand]
        
        # Stage 2 & 3 (Back to GPU)
        candidates_names = [self.mid2name.get(m, "Unknown") for m in candidates_mids]
        
        q_enc = self.s1_tok([question]*len(candidates_mids), return_tensors='pt', padding=True, truncation=True).to(self.device)
        p_enc = self.s1_tok([path_str]*len(candidates_mids), return_tensors='pt', padding=True, truncation=True).to(self.device)
        e_enc = self.s1_tok(candidates_names, return_tensors='pt', padding=True, truncation=True).to(self.device)
        
        scores = self.s2_model(q_enc['input_ids'], q_enc['attention_mask'],
                               p_enc['input_ids'], p_enc['attention_mask'],
                               e_enc['input_ids'], e_enc['attention_mask'])
        
        top_k_val_s2 = min(20, len(candidates_mids))
        top_idx_s2 = torch.topk(scores, top_k_val_s2).indices.tolist()
        final_candidates_mids = [candidates_mids[i] for i in top_idx_s2]
        final_candidates_names = [candidates_names[i] for i in top_idx_s2]
        
        qs = [question] * len(final_candidates_mids)
        enc = self.s3_tok(qs, final_candidates_names, return_tensors='pt', padding=True, truncation=True).to(self.device)
        logits = self.s3_model(**enc).logits.squeeze(-1)
        final_idx = torch.argmax(logits).item()
        
        return final_candidates_mids[final_idx]

class MasterEvaluator:
    def __init__(self, exp15_ckpt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Master] Device: {self.device}")
        
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Exp 15
        base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.exp15_agent = STRLAgent(base).to(self.device)
        self.exp15_agent.load_state_dict(torch.load(exp15_ckpt, map_location=self.device))
        self.exp15_agent.eval()
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
        
        # Exp 7
        self.exp7_agent = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.exp7_agent.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=self.device))
        self.exp7_agent.eval()
        
        # Exp 9
        self.exp9_agent = RLConstraintAgent(ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id))).to(self.device)
        self.exp9_agent.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=self.device))
        self.exp9_agent.eval()

        self.cds = CascadingDustSeparator(self.device)
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

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

    def run_benchmark(self, model_key, samples):
        print(f"\n[Bench] Running {model_key.upper()} with Optimized CDS...")
        stats = {'hit1': 0, 'hit_n': 0, 'ents': 0, 'count': 0}
        
        for s in tqdm(samples, desc=f"Eval {model_key}"):
            inputs = self.tokenizer(s['q'], return_tensors="pt", padding=True, truncation=True).to(self.device)
            current = {s['te']}
            path_log = []
            
            if model_key == 'exp7':
                out = self.exp7_agent(inputs['input_ids'], inputs['attention_mask'])
                for h in range(4):
                    if torch.sigmoid(out['stop_logits'][0, h]).item() < 0.5: break
                    rel = self.id2rel[torch.argmax(out['rel_logits'][0, h]).item()]
                    path_log.append(rel); current = self.kg_lookup(current, [rel])
                    if not current: break
            elif model_key == 'exp9':
                action_logits, _, rel_logits, dom_logits = self.exp9_agent(inputs['input_ids'], inputs['attention_mask'])
                pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
                domain_name = self.id2dom[pred_dom_id]
                for h in range(4):
                    action = torch.argmax(action_logits[0, h]).item()
                    if action == 3: break
                    if action == 0: active = [self.id2rel[torch.argmax(rel_logits[0, h]).item()]]
                    elif action == 1: active = [self.id2rel[rid] for rid in torch.topk(rel_logits[0, h], 5).indices.tolist()]
                    else: active = [r for r in self.id2rel.values() if domain_name in r]
                    path_log.append(active[0]); current = self.kg_lookup(current, active)
                    if not current: break
            elif model_key == 'exp15':
                fwd = self.exp15_agent(inputs['input_ids'], inputs['attention_mask'])
                for h in range(4):
                    action = torch.argmax(fwd['action_logits'][0, h]).item()
                    if action == 3: break
                    sims = torch.mv(self.rel_emb_bank.all(), fwd['hop_reprs'][0, h])
                    k = {0:5, 1:10, 2:50}.get(action, 5)
                    top_k = torch.topk(sims, k).indices.tolist()
                    rel = self.id2rel[top_k[0]]
                    path_log.append(rel); current = self.kg_lookup(current, [self.id2rel[rid] for rid in top_k])
                    if not current: break

            path_str = " -> ".join(path_log)
            stats['count'] += 1; stats['ents'] += len(current)
            if any(mid in s['gold'] for mid in current):
                stats['hit_n'] += 1
                predicted_mid = self.cds.separate_dust(s['q'], path_str, current)
                if predicted_mid in s['gold']: stats['hit1'] += 1
            
        return {
            'hit1': (stats['hit1']/stats['count'])*100 if stats['count'] else 0,
            'hit_n': (stats['hit_n']/stats['count'])*100 if stats['count'] else 0,
            'avg_ents': stats['ents']/stats['count'] if stats['count'] else 0
        }

def run_master_benchmark():
    evaluator = MasterEvaluator(os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_19.pt'))
    with open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8') as f:
        data = json.load(f); samples = []
        for item in data:
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            samples.append({'q': item['question'], 'te': path[0][0].replace("ns:", ""), 'gold': set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))})
    
    final_results = {}
    for key in ['exp7', 'exp9', 'exp15']:
        final_results[key] = evaluator.run_benchmark(key, samples)
    
    print("\n" + "="*60 + "\nFINAL TRIPLE-MODEL BENCHMARK (VRAM OPTIMIZED)\n" + "="*60)
    for k, res in final_results.items():
        print(f"[{k.upper()}]\n  Hit@1 Accuracy: {res['hit1']:.2f}%\n  Hit@N Recall:   {res['hit_n']:.2f}%\n  Avg Entities:   {res['avg_ents']:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_master_benchmark()
