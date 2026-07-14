"""
================================================================================
EXPERIMENT 7: SCALED UNIFIED PLANNER (STEP-BY-STEP WALKTHROUGH)
================================================================================
Architecture: 
- RoBERTa-Large Encoder
- Multi-Hop Transformer Planner
- Adjacency-based KG Execution

This script demonstrates how a question is processed through a fixed-path 
reasoning pipeline.
================================================================================
"""

import os, sys, json, torch
import torch.nn.functional as F
from transformers import RobertaTokenizer

# Ensure we can import from the root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner

class Exp7Walkthrough:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Initializing Exp 7 Pipeline on {self.device}...")
        
        # 1. LOAD MAPPINGS
        # These files map numerical IDs predicted by the model to real strings.
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. LOAD KNOWLEDGE GRAPH
        # This is the 4GB binary graph we built in Phase 3.
        kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
        self.kg = torch.load(kg_path, map_location='cpu')
        
        # 3. LOAD ENTITY NAMES
        # For mapping MIDs (m.0xxx) to names (e.g., "Inception").
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

        # 4. INITIALIZE MODEL
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        self.model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        
        # Load pre-trained weights
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt')
        
        print(f"[Init] Loading weights from {checkpoint_path}...")
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print("[Warning] Checkpoint not found. Running with random weights.")
            
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def run_inference(self, question, topic_mid):
        """
        Walks through the inference process for a single question.
        """
        print(f"\n[STEP 1: TEXT PROCESSING]")
        print(f"Input Question: \"{question}\"")
        
        # Tokenization converts text into a format the model can process.
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # --- FORWARD PASS ---
        with torch.no_grad():
            # The model encodes the question and runs the internal transformer planner.
            out = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # [STEP 2: DOMAIN PREDICTION]
        # The model predicts the general category (e.g., "film", "people").
        dom_logits = out['domain_logits']
        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]
        print(f"Predicted Domain: {domain_name}")

        # [STEP 3: MULTI-HOP PLANNING]
        # The model predicts a sequence of relations. 
        # Experiment 7 follows a single path (greedy decoding).
        current_entities = {topic_mid}
        path_taken = []

        for h in range(self.model.max_hops):
            print(f"\n--- Reasoning Hop {h+1} ---")
            
            # Predict Relation for this hop
            rel_logits = out['rel_logits'][0, h]
            top_rel_id = torch.argmax(rel_logits).item()
            relation_name = self.id2rel[top_rel_id]
            
            # Predict if we should STOP
            stop_logit = out['stop_logits'][0, h]
            keep_going_prob = torch.sigmoid(stop_logit).item()
            
            print(f"  Action: Predict Relation '{relation_name}'")
            print(f"  Confidence to continue: {keep_going_prob:.4f}")

            if keep_going_prob < 0.5:
                print(f"  -> Model reached a logical stop point.")
                break
            
            # [STEP 4: KG EXECUTION]
            # We look up the selected relation in the Knowledge Graph.
            next_entities = set()
            for ent in current_entities:
                # Check neighbors connected via the predicted relation
                for r, tgt in self.kg['forward'].get(ent, []):
                    if r == relation_name: next_entities.add(tgt)
                for r, src in self.kg['backward'].get(ent, []):
                    if r == relation_name: next_entities.add(src)
            
            if not next_entities:
                print(f"  !! Found no paths for '{relation_name}' in KG. Stopping.")
                break
                
            current_entities = next_entities
            path_taken.append(relation_name)
            
            # Sample output for the user
            names = [self.mid2name.get(e, e) for e in list(current_entities)[:3]]
            print(f"  -> Currently at {len(current_entities)} entities: {names}...")

        print("\n[STEP 5: FINAL ANSWER]")
        print(f"Path: {' -> '.join(path_taken)}")
        final_answers = [self.mid2name.get(e, e) for e in list(current_entities)[:10]]
        print(f"Answer List: {final_answers}")

if __name__ == "__main__":
    # Example: Nicholas S. Zeppos coached which university?
    walker = Exp7Walkthrough()
    walker.run_inference("Who is the mascot of the university where Nicholas Zeppos was chancellor?", "m.02vymvp")
