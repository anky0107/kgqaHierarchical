"""
STRATEGY 3: PICKLE PROTOCOL 5
Fastest RAM loading. Uses Python's optimized serialization format.
================================================================================
"""

import os, sys, json, torch, time, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer

"""
ARCHITECTURE DETAILS (Exp 7):
- Encoder: Roberta-Large (Transformer base)
- Projection Head: Maps 1024-dim Roberta output to 512-dim internal space.
- Planner: Multi-layer Transformer that calculates hop-specific reasoning context.
- Relation Head: Outputs probability distribution over all KG relations.
- Stop Head: Sigmoid-based confidence to end multi-hop search.
"""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner

class Exp7Pickle:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- LOADING PHASE ---
        print("\n[PHASE] Starting Pickle v5 Load...")
        start_time = time.time()
        
        # 1. Load Mappings
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. LOAD KG (Using Pickle v5)
        print("  -> Loading 4GB augmented_kg_v5.pkl into RAM...")
        kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_v5.pkl')
        with open(kg_path, 'rb') as f:
            self.kg = pickle.load(f)
        
        load_duration = time.time() - start_time
        print(f"[METRIC] Total Loading Time: {load_duration:.4f} seconds\n")

        # 3. Model Initialization & Weight Loading
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        print(f"[Init] Initializing ScaledUnifiedPlanner ({num_rel} relations)...")
        self.model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        
        # Loading best epoch weights
        checkpoint_path = os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt')
        print(f"[Init] Loading fine-tuned weights from: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        print("[Init] Pipeline fully ready with best epoch weights.\n")

    def run(self, question, topic_mid):
        print(f"Running Inference for: {question}")
        start_inf = time.time()
        
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # KG Traversal Logic
        current = {topic_mid}
        rel_logits = out['rel_logits'][0, 0]
        top_rel = self.id2rel[torch.argmax(rel_logits).item()]
        
        next_ents = set()
        for ent in current:
            for r, tgt in self.kg['forward'].get(ent, []):
                if r == top_rel: next_ents.add(tgt)
            for r, src in self.kg['backward'].get(ent, []):
                if r == top_rel: next_ents.add(src)
        
        inf_duration = time.time() - start_inf
        print(f"[METRIC] Inference Traversal Time: {inf_duration:.4f} seconds")
        print(f"Results: {list(next_ents)[:3]}...")

if __name__ == "__main__":
    app = Exp7Pickle()
    app.run("Where did the 'Country Nation World Tour' artist go to college?", "m.010qhfmm")
