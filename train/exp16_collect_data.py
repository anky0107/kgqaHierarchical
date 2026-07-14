import os, sys, json, torch, lmdb, pickle
from tqdm import tqdm
from transformers import RobertaTokenizer

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from inference_pipeline.model import ScaledUnifiedPlanner
from utils.sparql_parser import find_reasoning_path

class CDSDataCollector:
    def __init__(self, exp15_ckpt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Collector] Initializing with {self.device}")
        
        # Load Exp15 Agent
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        
        base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.agent = STRLAgent(base).to(self.device)
        self.agent.load_state_dict(torch.load(exp15_ckpt, map_location=self.device))
        self.agent.eval()
        
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
        self.rel_emb_bank.eval()
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # KG for traversal
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
        for item in tqdm(data, desc=f"Collecting {os.path.basename(input_file)}"):
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            
            q = item['question']
            te = path[0][0].replace("ns:", "")
            gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
            
            # Run Exp15 Traversal
            inputs = self.tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(self.device)
            fwd = self.agent(inputs['input_ids'], inputs['attention_mask'])
            
            current = {te}
            execution_log = []
            
            for h in range(4):
                action = torch.argmax(fwd['action_logits'][0, h]).item()
                if action == 3: break
                hop_repr = fwd['hop_reprs'][0, h]
                
                # Semantic Beam
                all_embs = self.rel_emb_bank.all()
                sims = torch.mv(all_embs, hop_repr)
                k = {0:5, 1:10, 2:50}.get(action, 5)
                
                # Filter by reachability
                reachable_rels = set()
                with self.env.begin() as txn:
                    for mid in current:
                        f_data = txn.get(f"f:{mid}".encode('utf-8'))
                        if f_data:
                            for r, _ in pickle.loads(f_data): 
                                if r in self.rel2id: reachable_rels.add(self.rel2id[r])
                        b_data = txn.get(f"b:{mid}".encode('utf-8'))
                        if b_data:
                            for r, _ in pickle.loads(b_data):
                                if r in self.rel2id: reachable_rels.add(self.rel2id[r])
                
                top_k = torch.topk(sims, k).indices.tolist()
                valid_beam = [rid for rid in top_k if rid in reachable_rels]
                active = [self.id2rel[rid] for rid in (valid_beam or top_k[:k])]
                execution_log.append(active)
                
                next_ents = self.kg_lookup(current, active)
                if not next_ents: break
                current = next_ents
            
            # Final Beam Candidates with Names
            candidates = []
            for mid in current:
                name = self.mid2name.get(mid, "Unknown")
                candidates.append({'mid': mid, 'name': name, 'is_gold': mid in gold})
            
            results.append({
                'question': q,
                'path': execution_log,
                'candidates': candidates
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"[Collector] Saved {len(results)} samples to {output_file}")

if __name__ == "__main__":
    collector = CDSDataCollector(os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_19.pt'))
    # Collect a subset for training the ranker (faster for now)
    collector.collect(os.path.join(ROOT, 'data/cwq_train.json'), os.path.join(ROOT, 'data/exp16_cds_train.json'), limit=2000)
    collector.collect(os.path.join(ROOT, 'data/cwq_dev.json'), os.path.join(ROOT, 'data/exp16_cds_dev.json'))
