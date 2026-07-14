import os, sys, json, torch, torch.nn.functional as F, lmdb, pickle
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.sparql_parser import find_reasoning_path

class ZeroShotRobertaAgent:
    def __init__(self, device):
        self.device = device
        print(f"[ZeroShot] Initializing raw roberta-large...")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large').to(device)
        self.model.eval()
        
        # Load mappings
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        # Pre-compute relation embeddings
        print("[ZeroShot] Pre-computing embeddings for all relations...")
        self.rel_names = [self.id2rel[i] for i in range(len(self.id2rel))]
        # Simplify relation names for better matching: e.g., 'award.award_honor.award_winner' -> 'award honor award winner'
        self.clean_rels = [r.replace('.', ' ').replace('_', ' ') for r in self.rel_names]
        
        self.rel_embs = self._encode_batch(self.clean_rels)
        
        # KG
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    @torch.no_grad()
    def _encode_batch(self, texts, batch_size=32):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            out = self.model(**enc).last_hidden_state[:, 0, :]
            embs.append(out.cpu())
        return torch.cat(embs, dim=0)

    def kg_lookup(self, entities, rel_names):
        next_entities = set()
        with self.env.begin() as txn:
            for ent in entities:
                f_data = txn.get(f"f:{ent}".encode())
                if f_data:
                    for r, tgt in pickle.loads(f_data):
                        if r in rel_names: next_entities.add(tgt)
                b_data = txn.get(f"b:{ent}".encode())
                if b_data:
                    for r, src in pickle.loads(b_data):
                        if r in rel_names: next_entities.add(src)
        return next_entities

    def find_reachable_rels(self, entities):
        rels = set()
        with self.env.begin() as txn:
            for ent in entities:
                f_data = txn.get(f"f:{ent}".encode())
                if f_data:
                    for r, _ in pickle.loads(f_data): rels.add(r)
                b_data = txn.get(f"b:{ent}".encode())
                if b_data:
                    for r, _ in pickle.loads(b_data): rels.add(r)
        return rels

    @torch.no_grad()
    def run_eval(self, dev_file):
        print(f"[ZeroShot] Evaluating on full dev set: {dev_file}...")
        with open(dev_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {'hit1': 0, 'count': 0}
        
        for item in tqdm(data, desc="Zero-Shot RoBERTa"):
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            
            q = item['question']
            te = path[0][0].replace("ns:", "")
            gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
            num_hops = len(path)
            
            # Encode question
            q_enc = self.tokenizer(q, return_tensors='pt', padding=True, truncation=True).to(self.device)
            q_emb = self.model(**q_enc).last_hidden_state[:, 0, :].cpu()
            
            # --- ZERO SHOT REASONING ---
            current = {te}
            for h in range(num_hops):
                reachable = self.find_reachable_rels(current)
                if not reachable: break
                
                # Filter our pre-computed embeddings for reachable relations
                reachable_list = list(reachable)
                reachable_indices = [self.rel2id[r] for r in reachable_list if r in self.rel2id]
                if not reachable_indices: break
                
                r_embs = self.rel_embs[reachable_indices]
                sims = F.cosine_similarity(q_emb, r_embs)
                
                # Greedy: pick top-1 relation
                best_rel_idx = torch.argmax(sims).item()
                best_rel = reachable_list[best_rel_idx]
                
                current = self.kg_lookup(current, [best_rel])
                if not current: break
            
            if not current:
                stats['count'] += 1
                continue
                
            # --- ZERO SHOT SELECTION ---
            mids = list(current)
            names = [self.mid2name.get(m, m) for m in mids]
            
            # Cap to 500 candidates for speed in zero-shot eval
            if len(names) > 500:
                names = names[:500]
                mids = mids[:500]
            
            e_embs = self._encode_batch(names)
            e_sims = F.cosine_similarity(q_emb, e_embs)
            best_e_idx = torch.argmax(e_sims).item()
            pred_mid = mids[best_e_idx]
            
            stats['count'] += 1
            if pred_mid in gold:
                stats['hit1'] += 1
        
        print("\n" + "="*50)
        print("ZERO-SHOT ROBERTA-LARGE FULL EVAL")
        print("="*50)
        print(f"Total Questions: {stats['count']}")
        print(f"Hit@1 Accuracy:  {(stats['hit1']/stats['count'])*100:.2f}%")
        print("="*50)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ZeroShotRobertaAgent(device)
    agent.run_eval(os.path.join(ROOT, 'data/cwq_dev.json'))
