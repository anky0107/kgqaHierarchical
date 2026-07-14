import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from utils.sparql_parser import find_reasoning_path

print("Starting lightweight evaluation...")

class NaturalEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        print("Loading Exp7 Model...")
        self.model7 = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.model7.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=self.device))
        self.model7.eval()
        
        print("Loading Exp9 Model...")
        base9 = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.model9 = RLConstraintAgent(base9).to(self.device)
        self.model9.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=self.device))
        self.model9.eval()
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        print("Models loaded.")

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
    def run_exp7(self, question, topic_mid):
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model7(inputs['input_ids'], inputs['attention_mask'])
        current = {topic_mid}
        for h in range(self.model7.max_hops):
            if torch.sigmoid(out['stop_logits'][0, h]).item() < 0.5: break
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
            if action == 3: break
            if action == 0: active = [self.id2rel[torch.argmax(rel_logits[0, h]).item()]]
            elif action == 1: active = [self.id2rel[rid] for rid in torch.topk(rel_logits[0, h], 5).indices.tolist()]
            elif action == 2: active = [r for r in self.id2rel.values() if domain_name in r]
            next_ents = self.kg_lookup(current, active)
            if not next_ents: break
            current = next_ents
        return current

def main():
    try:
        evaluator = NaturalEvaluator()
        train_path = os.path.join(ROOT, 'data/cwq_train.json')
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        limit = 2
        samples = []
        for item in data:
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            te = path[0][0][3:] if path[0][0].startswith("ns:") else path[0][0]
            samples.append({'q': item['question'], 'te': te, 'gold': set(a['answer_id'] for a in item['answers'])})
            if len(samples) >= limit: break
            
        print(f"Evaluating {len(samples)} samples...")
        results = {'exp7': {'hits': 0, 'ents': 0, 'count': 0}, 'exp9': {'hits': 0, 'ents': 0, 'count': 0}}
        
        for i, s in enumerate(samples):
            print(f"Sample {i+1}/{limit}")
            # Exp 7
            ans7 = evaluator.run_exp7(s['q'], s['te'])
            if any(a in s['gold'] for a in ans7): results['exp7']['hits'] += 1
            results['exp7']['ents'] += len(ans7)
            results['exp7']['count'] += 1
            
            # Exp 9
            ans9 = evaluator.run_exp9(s['q'], s['te'])
            if any(a in s['gold'] for a in ans9): results['exp9']['hits'] += 1
            results['exp9']['ents'] += len(ans9)
            results['exp9']['count'] += 1
            
        print("Eval finished. Saving...")
        with open(os.path.join(ROOT, 'inference_pipeline/eval_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print("Done.")
        with open(os.path.join(ROOT, 'inference_pipeline/finished.txt'), 'w') as f:
            f.write("Evaluation complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
