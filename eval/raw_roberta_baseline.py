import os, sys, json, torch, torch.nn.functional as F, lmdb, pickle
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.sparql_parser import find_reasoning_path

class RawRobertaBaseline:
    def __init__(self, device):
        self.device = device
        print(f"[Baseline] Loading raw roberta-large (pre-trained only)...")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large').to(device)
        self.model.eval()
        
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        # KG for beam generation (using Exp 15 logic)
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def kg_lookup(self, entities, rels):
        next_entities = set()
        with self.env.begin() as txn:
            for ent in entities:
                for rel in rels:
                    f_data = txn.get(f"f:{ent}".encode())
                    if f_data:
                        for r, tgt in pickle.loads(f_data):
                            if r == rel: next_entities.add(tgt)
                    b_data = txn.get(f"b:{ent}".encode())
                    if b_data:
                        for r, src in pickle.loads(b_data):
                            if r == rel: next_entities.add(src)
        return next_entities

    @torch.no_grad()
    def get_semantic_score(self, question, entity_names):
        """Zero-shot semantic similarity using raw RoBERTa [CLS] embeddings."""
        q_enc = self.tokenizer(question, return_tensors='pt', padding=True, truncation=True).to(self.device)
        q_emb = self.model(**q_enc).last_hidden_state[:, 0, :] # [1, 1024]
        
        scores = []
        # Batch entity encoding for speed
        batch_size = 32
        for i in range(0, len(entity_names), batch_size):
            batch = entity_names[i:i+batch_size]
            e_enc = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            e_embs = self.model(**e_enc).last_hidden_state[:, 0, :] # [B, 1024]
            
            sims = F.cosine_similarity(q_emb, e_embs)
            scores.extend(sims.cpu().tolist())
        return scores

    def evaluate(self, dev_file, limit=500):
        print(f"[Baseline] Evaluating on {dev_file} (limit={limit})...")
        with open(dev_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {'hit1': 0, 'hit_n': 0, 'count': 0}
        for item in tqdm(data[:limit], desc="Raw RoBERTa Eval"):
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            
            q = item['question']
            te = path[0][0].replace("ns:", "")
            gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
            
            # 1. Generate Beam (use gold relations to isolate the "Selection" problem)
            # Or use a fixed agent? The user said "just take roberta model".
            # I'll use the gold path to see how well RoBERTa can pick the answer from the gold beam.
            # This is the fairest test of "Selection" capability.
            current = {te}
            for _, rel, _, _ in path:
                current = self.kg_lookup(current, [rel])
                if not current: break
            
            if not current: continue
            
            stats['count'] += 1
            if any(mid in gold for mid in current):
                stats['hit_n'] += 1
                
                # 2. Selection using raw RoBERTa
                mids = list(current)
                # Optimization: Cap candidates to 1000 for speed
                if len(mids) > 1000:
                    import random
                    # Keep gold if present in the pool we sample
                    g_in_c = [m for m in mids if m in gold]
                    o_in_c = [m for m in mids if m not in gold]
                    mids = g_in_c + random.sample(o_in_c, min(1000 - len(g_in_c), len(o_in_c)))
                
                names = [self.mid2name.get(m, m) for m in mids]
                
                scores = self.get_semantic_score(q, names)
                best_idx = scores.index(max(scores))
                pred_mid = mids[best_idx]
                
                if pred_mid in gold:
                    stats['hit1'] += 1
        
        print("\n" + "="*50)
        print("RAW ROBERTA-LARGE BASELINE (ZERO-SHOT)")
        print("="*50)
        print(f"Total Questions: {stats['count']}")
        print(f"Hit@N Recall:    {(stats['hit_n']/stats['count'])*100:.2f}% (Gold Path)")
        print(f"Hit@1 Accuracy:  {(stats['hit1']/stats['count'])*100:.2f}%")
        print("="*50)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = RawRobertaBaseline(device)
    baseline.evaluate(os.path.join(ROOT, 'data/cwq_dev.json'))
