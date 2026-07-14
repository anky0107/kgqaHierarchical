"""
STRATEGY 1: BASELINE (torch.load)
Loads the entire 4GB KG into RAM at startup.
================================================================================
"""

import os, sys, json, torch, time
import torch.nn.functional as F
from transformers import RobertaTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner

class Exp7Baseline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- LOADING PHASE ---
        print("\n[PHASE] Starting Baseline Load...")
        start_time = time.time()
        
        # 1. Load Mappings
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. LOAD KG (The slow part)
        print("  -> Loading 4GB augmented_kg.pt into RAM...")
        kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
        self.kg = torch.load(kg_path, map_location='cpu')
        
        load_duration = time.time() - start_time
        print(f"[METRIC] Total Loading Time: {load_duration:.4f} seconds\n")

        # 3. Model Prep
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        self.model = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def run(self, question, topic_mid):
        print(f"Running Inference for: {question}")
        start_inf = time.time()
        
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # KG Traversal Logic (Standard)
        current = {topic_mid}
        rel_logits = out['rel_logits'][0, 0] # First hop
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
    app = Exp7Baseline()
    app.run("Where did the 'Country Nation World Tour' artist go to college?", "m.010qhfmm")
