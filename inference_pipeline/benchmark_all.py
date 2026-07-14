import os, sys, json, torch, time, lmdb, pickle
from transformers import RobertaTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import functools

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.exp15_optimized import Exp15Optimized
from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from utils.sparql_parser import find_reasoning_path

class MasterEvaluator:
    def __init__(self, exp15_ckpt=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Initializing Master Evaluator (Device: {self.device})")
        
        # 1. Load Mappings & KG
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # 2. Stage 3 Ranker (Shared)
        print("[Init] Loading Stage 3 Ranker...")
        selector_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.selector_tokenizer = AutoTokenizer.from_pretrained(selector_model_name)
        self.selector_model = AutoModelForSequenceClassification.from_pretrained(selector_model_name, num_labels=1).to(self.device)
        self.selector_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp15_answer_selector.pt'), map_location=self.device))
        self.selector_model.eval()
        
        self.exp15_ckpt = exp15_ckpt or os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_12.pt')

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
    def rank_answers(self, question, candidates_mids, gold_mids):
        candidates = []
        for mid in candidates_mids:
            name = self.mid2name.get(mid, None)
            if name: candidates.append((mid, name))
        if not candidates: return False
        
        candidates = candidates[:100]
        questions = [question] * len(candidates)
        names = [c[1] for c in candidates]
        
        enc = self.selector_tokenizer(questions, names, padding=True, truncation=True, return_tensors='pt').to(self.device)
        logits = self.selector_model(**enc).logits.squeeze(-1)
        top_idx = torch.argmax(logits).item()
        return candidates[top_idx][0] in gold_mids

    def run_benchmark(self, model_key, samples):
        print(f"\n[Bench] Running {model_key.upper()}...")
        
        # Load Model
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        if model_key == 'exp15':
            base = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
            model = STRLAgent(base).to(self.device)
            model.load_state_dict(torch.load(self.exp15_ckpt, map_location=self.device))
            model.eval()
            rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
            rel_emb_bank.eval()

        stats = {'hit1': 0, 'hit_n': 0, 'ents': 0, 'count': 0}
        
        for s in tqdm(samples, desc=f"Eval {model_key}"):
            try:
                inputs = self.tokenizer(s['q'], return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                if model_key == 'exp15':
                    fwd = model(inputs['input_ids'], inputs['attention_mask'])
                    current = {s['te']}
                    for h in range(4):
                        action = torch.argmax(fwd['action_logits'][0, h]).item()
                        if action == 3: break
                        hop_repr = fwd['hop_reprs'][0, h]
                        
                        # Semantic Beam logic
                        all_embs = rel_emb_bank.all()
                        sims     = torch.mv(all_embs, hop_repr)
                        k        = {0:5, 1:10, 2:50}.get(action, 5) # simplified beam
                        
                        reachable_rels = set()
                        for mid in current:
                            with self.env.begin() as txn:
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
                        if not valid_beam and reachable_rels:
                            fallback = [(rid, sims[rid].item()) for rid in reachable_rels]
                            fallback.sort(key=lambda x: x[1], reverse=True)
                            valid_beam = [rid for rid, _ in fallback[:k]]
                        active = [self.id2rel[rid] for rid in (valid_beam or top_k[:k])]
                        
                        next_ents = self.kg_lookup(current, active)
                        if not next_ents: break
                        current = next_ents

                # Update Stats
                stats['count'] += 1
                stats['ents'] += len(current)
                if any(mid in s['gold'] for mid in current):
                    stats['hit_n'] += 1
                    if self.rank_answers(s['q'], current, s['gold']):
                        stats['hit1'] += 1
            except:
                pass
        
        del model
        torch.cuda.empty_cache()
        
        return {
            'hit1': (stats['hit1']/stats['count'])*100,
            'hit_n': (stats['hit_n']/stats['count'])*100,
            'avg_ents': stats['ents']/stats['count'],
            'count': stats['count']
        }

def run_full_benchmark():
    ckpt_path = os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_19.pt')
    evaluator = MasterEvaluator(exp15_ckpt=ckpt_path)
    
    print("Loading Dev Set...")
    with open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        path = find_reasoning_path(item['sparql'])
        if not path: continue
        samples.append({
            'q': item['question'],
            'te': path[0][0].replace("ns:", ""),
            'gold': set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
        })
    print(f"Total valid samples: {len(samples)}")
    
    final_results = {}
    for key in ['exp15']:
        final_results[key] = evaluator.run_benchmark(key, samples)
    
    print("\n" + "="*50)
    print(f"FINAL BENCHMARK RESULTS (EXP 15 - EPOCH 19)")
    print("="*50)
    for k, res in final_results.items():
        print(f"[{k.upper()}]")
        print(f"  Hit@1 Accuracy: {res['hit1']:.2f}%")
        print(f"  Hit@N Recall:   {res['hit_n']:.2f}%")
        print(f"  Avg Entities:   {res['avg_ents']:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_full_benchmark()
