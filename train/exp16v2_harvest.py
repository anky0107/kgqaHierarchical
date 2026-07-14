"""
Exp 16 v2: Multi-Model Harvester
=================================
Collects hard-negative training data from ALL THREE models (Exp 7, 9, 15).
This produces a truly generalized CDS that is not tied to any single model's beam distribution.
"""
import os, sys, json, torch, lmdb, pickle
from tqdm import tqdm
from transformers import RobertaTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from utils.sparql_parser import find_reasoning_path

class MultiModelHarvester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Harvester] Initializing on {self.device}...")

        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        # Load Exp 7
        print("[Harvester] Loading Exp 7...")
        self.exp7 = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.exp7.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=self.device))
        self.exp7.eval()

        # Load Exp 9
        print("[Harvester] Loading Exp 9...")
        self.exp9 = RLConstraintAgent(ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id))).to(self.device)
        self.exp9.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=self.device))
        self.exp9.eval()

        # Load Exp 15
        print("[Harvester] Loading Exp 15...")
        base15 = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.exp15 = STRLAgent(base15).to(self.device)
        self.exp15.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_19.pt'), map_location=self.device))
        self.exp15.eval()
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)

        # KG & entity names
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

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
    def get_beam(self, model_key, inputs, te):
        current = {te}
        path_log = []
        if model_key == 'exp7':
            out = self.exp7(inputs['input_ids'], inputs['attention_mask'])
            for h in range(4):
                if torch.sigmoid(out['stop_logits'][0, h]).item() < 0.5: break
                rel = self.id2rel[torch.argmax(out['rel_logits'][0, h]).item()]
                path_log.append(rel)
                current = self.kg_lookup(current, [rel])
                if not current: break
        elif model_key == 'exp9':
            action_logits, _, rel_logits, dom_logits = self.exp9(inputs['input_ids'], inputs['attention_mask'])
            dom_name = self.id2dom[torch.argmax(dom_logits, dim=-1).item()]
            for h in range(4):
                action = torch.argmax(action_logits[0, h]).item()
                if action == 3: break
                if action == 0: active = [self.id2rel[torch.argmax(rel_logits[0, h]).item()]]
                elif action == 1: active = [self.id2rel[rid] for rid in torch.topk(rel_logits[0, h], 5).indices.tolist()]
                else: active = [r for r in self.id2rel.values() if dom_name in r]
                path_log.append(active[0])
                current = self.kg_lookup(current, active)
                if not current: break
        elif model_key == 'exp15':
            fwd = self.exp15(inputs['input_ids'], inputs['attention_mask'])
            for h in range(4):
                action = torch.argmax(fwd['action_logits'][0, h]).item()
                if action == 3: break
                sims = torch.mv(self.rel_emb_bank.all(), fwd['hop_reprs'][0, h])
                k = {0:5, 1:10, 2:50}.get(action, 5)
                top_k = torch.topk(sims, k).indices.tolist()
                path_log.append(self.id2rel[top_k[0]])
                current = self.kg_lookup(current, [self.id2rel[rid] for rid in top_k])
                if not current: break
        return current, " -> ".join(path_log)

    def harvest(self, input_file, output_file, max_samples=None):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if max_samples: data = data[:max_samples]

        results = []
        for item in tqdm(data, desc="Multi-Model Harvest"):
            try:
                path = find_reasoning_path(item['sparql'])
                if not path: continue
                q = item['question']
                te = path[0][0].replace("ns:", "")
                gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
                inputs = self.tokenizer(q, return_tensors='pt', padding=True, truncation=True).to(self.device)

                # Collect from all 3 models
                combined_beam = {}
                for model_key in ['exp7', 'exp9', 'exp15']:
                    beam, path_str = self.get_beam(model_key, inputs, te)
                    for mid in beam:
                        combined_beam[mid] = path_str  # last write wins (prefer exp15 path)

                candidates = []
                for mid, path_str in combined_beam.items():
                    candidates.append({
                        'mid': mid,
                        'name': self.mid2name.get(mid, "Unknown"),
                        'is_gold': mid in gold,
                        'path': path_str
                    })

                golds = [c for c in candidates if c['is_gold']]
                negs = [c for c in candidates if not c['is_gold']]

                if not golds: continue
                
                # Keep max 30 negatives to prevent massive 28GB file explosion
                import random
                candidates = golds + random.sample(negs, min(30, len(negs)))

                results.append({
                    'question': q,
                    'path': list(combined_beam.values())[-1] if combined_beam else "",
                    'candidates': candidates
                })
            except: pass

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"[Harvester] Saved {len(results)} samples → {output_file}")

if __name__ == "__main__":
    harvester = MultiModelHarvester()
    harvester.harvest(
        os.path.join(ROOT, 'data/cwq_train.json'),
        os.path.join(ROOT, 'data/exp16v2_cds_train.json')
    )
