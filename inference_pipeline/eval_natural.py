import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from utils.sparql_parser import find_reasoning_path

class NaturalEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Initializing Evaluator (Device: {self.device})...")
        
        # 1. Load Mappings
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. Load KG (LMDB)
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        # 3. Load Models
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        # Exp 7
        self.model7 = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.model7.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=self.device))
        self.model7.eval()
        
        # Exp 9
        base9 = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.model9 = RLConstraintAgent(base9).to(self.device)
        self.model9.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=self.device))
        self.model9.eval()
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Stage 3: Answer Selector
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        selector_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.selector_tokenizer = AutoTokenizer.from_pretrained(selector_model_name)
        self.selector_model = AutoModelForSequenceClassification.from_pretrained(selector_model_name, num_labels=1).to(self.device)
        self.selector_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp15_answer_selector.pt'), map_location=self.device))
        self.selector_model.eval()
        
        # Load Entity Names
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        print("[Init] Ready with Stage 3 Ranker.")

    def kg_lookup(self, entities, rels):
        next_entities = set()
        with self.env.begin() as txn:
            for ent in entities:
                for rel in rels:
                    # Forward
                    f_data = txn.get(f"f:{ent}".encode('utf-8'))
                    if f_data:
                        for r, tgt in pickle.loads(f_data):
                            if r == rel: next_entities.add(tgt)
                    # Backward
                    b_data = txn.get(f"b:{ent}".encode('utf-8'))
                    if b_data:
                        for r, src in pickle.loads(b_data):
                            if r == rel: next_entities.add(src)
        return next_entities

    @torch.no_grad()
    def run_exp7(self, question, topic_mid):
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model7(inputs['input_ids'], inputs['attention_mask'])
        
        current = {topic_mid}
        for h in range(self.model7.max_hops):
            stop_prob = torch.sigmoid(out['stop_logits'][0, h]).item()
            if stop_prob < 0.5: break
            
            top_rel = self.id2rel[torch.argmax(out['rel_logits'][0, h]).item()]
            next_ents = self.kg_lookup(current, [top_rel])
            if not next_ents: break
            current = next_ents
        return current

    @torch.no_grad()
    def run_exp9(self, question, topic_mid):
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        action_logits, _, rel_logits, dom_logits = self.model9(inputs['input_ids'], inputs['attention_mask'])
        
        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]
        
        current = {topic_mid}
        for h in range(4):
            action = torch.argmax(action_logits[0, h]).item()
            if action == 3: break # STOP
            
            if action == 0: # TIGHT
                active = [self.id2rel[torch.argmax(rel_logits[0, h]).item()]]
            elif action == 1: # MEDIUM
                top5 = torch.topk(rel_logits[0, h], 5).indices.tolist()
                active = [self.id2rel[rid] for rid in top5]
            elif action == 2: # LOOSE
                active = [r for r in self.id2rel.values() if domain_name in r]
            
            next_ents = self.kg_lookup(current, active)
            if not next_ents: break
            current = next_ents
        return current

    @torch.no_grad()
    def get_hit1(self, question, candidates_mids, gold_mids):
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
        best_mid = candidates[top_idx][0]
        return best_mid in gold_mids

def evaluate():
    evaluator = NaturalEvaluator()
    
    print("Loading Dev Data...")
    train_path = os.path.join(ROOT, 'data/cwq_dev.json')
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    limit = None # Evaluate on full set
    samples = []
    for item in data:
        path = find_reasoning_path(item['sparql'])
        if not path: continue
        te = path[0][0][3:] if path[0][0].startswith("ns:") else path[0][0]
        samples.append({
            'q': item['question'],
            'te': te,
            'gold': set(a['answer_id'] for a in item['answers'])
        })
        if limit is not None and len(samples) >= limit: break
        
    print(f"Evaluating {len(samples)} samples...")
    results = {
        'exp7': {'hits': 0, 'hit1': 0, 'ents': 0, 'count': 0},
        'exp9': {'hits': 0, 'hit1': 0, 'ents': 0, 'count': 0}
    }
    
    for s in tqdm(samples):
        # Exp 7
        try:
            ans7 = evaluator.run_exp7(s['q'], s['te'])
            if any(a in s['gold'] for a in ans7): results['exp7']['hits'] += 1
            if evaluator.get_hit1(s['q'], ans7, s['gold']): results['exp7']['hit1'] += 1
            results['exp7']['ents'] += len(ans7)
            results['exp7']['count'] += 1
        except: pass
        
        # Exp 9
        try:
            ans9 = evaluator.run_exp9(s['q'], s['te'])
            if any(a in s['gold'] for a in ans9): results['exp9']['hits'] += 1
            if evaluator.get_hit1(s['q'], ans9, s['gold']): results['exp9']['hit1'] += 1
            results['exp9']['ents'] += len(ans9)
            results['exp9']['count'] += 1
        except: pass

    print("\n" + "="*40)
    print("NATURAL PREDICTIONS EVALUATION (Dev Set)")
    print("="*40)
    summary = {}
    for k in ['exp7', 'exp9']:
        res = results[k]
        if res['count'] > 0:
            success_rate = (res['hits']/res['count'])*100
            hit1_rate = (res['hit1']/res['count'])*100
            avg_entities = res['ents']/res['count']
            print(f"[{k.upper()}]")
            print(f"  Success (Hit@N): {success_rate:.2f}%")
            print(f"  Accuracy (Hit@1): {hit1_rate:.2f}%")
            print(f"  Avg Entities:    {avg_entities:.2f}")
            print(f"  Samples:         {res['count']}")
            summary[k] = {
                'success_rate': success_rate,
                'avg_entities': avg_entities,
                'count': res['count']
            }
    print("="*40)
    
    with open(os.path.join(ROOT, 'inference_pipeline/eval_results.json'), 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    evaluate()
