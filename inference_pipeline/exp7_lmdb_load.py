"""
STRATEGY 2: LMDB (Memory-Mapped)
Near-instant startup. Data remains on disk and is accessed via memory-mapping.
================================================================================
"""

import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer

"""
ARCHITECTURE DETAILS (Exp 7):
- Encoder: Roberta-Large (Base model for text understanding).
- Projection Layer: Maps Roberta hidden state (1024) to Internal Dimension (512).
- Hop Transformer: A 2-layer transformer that 'thinks' about the question for 
  multiple steps, generating a unique context for each hop.
- Relation Head: A classifier over ~2000+ Freebase relations.
- Stop Head: A binary classifier to decide when reasoning is complete.
"""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner

class Exp7LMDB:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- LOADING PHASE ---
        print("\n[PHASE] Starting LMDB Load...")
        start_time = time.time()
        
        # 1. Load Mappings
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. OPEN LMDB (The lightning fast part)
        print("  -> Opening LMDB Environment (Zero-copy memory mapping)...")
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        # We don't load data yet, just open the interface
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        load_duration = time.time() - start_time
        print(f"[METRIC] Total Startup Time: {load_duration:.4f} seconds (Expected < 0.2s)\n")

        # 3. Model Initialization & Weight Loading
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        print(f"[Init] Initializing ScaledUnifiedPlanner ({num_rel} relations)...")
        self.model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        
        # Load the best epoch weights
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
        
        # KG Traversal Logic (Using LMDB)
        current = {topic_mid}
        rel_logits = out['rel_logits'][0, 0]
        top_rel = self.id2rel[torch.argmax(rel_logits).item()]
        
        next_ents = set()
        with self.env.begin() as txn:
            for ent in current:
                # Get Forward Edges from Disk
                f_key = f"f:{ent}".encode('utf-8')
                f_data = txn.get(f_key)
                if f_data:
                    neighbors = pickle.loads(f_data)
                    for r, tgt in neighbors:
                        if r == top_rel: next_ents.add(tgt)
                
                # Get Backward Edges from Disk
                b_key = f"b:{ent}".encode('utf-8')
                b_data = txn.get(b_key)
                if b_data:
                    neighbors = pickle.loads(b_data)
                    for r, src in neighbors:
                        if r == top_rel: next_ents.add(src)
        
        inf_duration = time.time() - start_inf
        print(f"[METRIC] Inference Traversal Time: {inf_duration:.4f} seconds")
        print(f"Results: {list(next_ents)[:3]}...")

if __name__ == "__main__":
    app = Exp7LMDB()
    app.run("Where did the 'Country Nation World Tour' artist go to college?", "m.010qhfmm")
