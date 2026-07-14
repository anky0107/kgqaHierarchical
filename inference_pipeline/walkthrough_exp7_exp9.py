"""
================================================================================
EXPERIMENT 7 & 9: STEP-BY-STEP INFERENCE WALKTHROUGH
================================================================================
This file provides a detailed walkthrough of the inference logic for two 
state-of-the-art KGQA models in this repository.

ARCHITECTURE OVERVIEW:
----------------------
1. EXP 7 (Scaled Unified Planner):
   - Encoder: RoBERTa-Large.
   - Planner: A multi-head transformer that refined the question embedding for 
     each hop (up to 4 hops).
   - Logic: Predicts a single relation at each hop and traverses the KG.

2. EXP 9 (RL Meta-Constraint Agent - RLMC):
   - Base: Reuses the Exp 7 Planner.
   - RL Layer: A Policy Head (trained via PPO) that decides the "Width" of 
     exploration at each hop.
   - Actions: 
     - TIGHT: Only follow the Top-1 predicted relation.
     - MEDIUM: Follow the Top-5 relations.
     - LOOSE: Follow all relations in the predicted Domain.
     - STOP: Halt the search immediately.

DATA FLOW:
----------
Question -> Tokenizer -> RoBERTa -> Hop-Specific Embeddings -> [Exp 9 Policy] -> 
KG Traversal -> Result Collection.
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
from train.exp9_rlmc import RLConstraintAgent

class DetailedInferenceWalkthrough:
    def __init__(self, mode='exp7'):
        """
        Initializes the inference pipeline.
        mode: 'exp7' or 'exp9'
        """
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- DATA LOADING ---
        # We need mappings to convert IDs from the model back into readable names.
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # Load the Knowledge Graph we just built in Phase 3
        kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
        self.kg = torch.load(kg_path, map_location='cpu')
        
        # Load Entity names for readable output
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

        # --- MODEL INITIALIZATION ---
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        # Both experiments use the ScaledUnifiedPlanner as the backbone
        self.base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        
        if mode == 'exp7':
            # Exp 7 uses only the base weights
            ckpt = os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt')
            self.base_model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.model = self.base_model
        else:
            # Exp 9 wraps the base model with an RL Policy head
            self.model = RLConstraintAgent(self.base_model).to(self.device)
            ckpt = os.path.join(ROOT, 'checkpoints/exp9_rlmc_best.pt')
            if os.path.exists(ckpt):
                self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def run(self, question, topic_mid):
        """
        Executes the step-by-step reasoning process.
        """
        print(f"\n[START] Processing Question: \"{question}\"")
        
        # 1. TOKENIZATION
        # Converts text into integers (IDs) that the transformer can understand.
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # 2. ENCODING & PLANNING
        # The model generates a sequence of "Plans" (one for each hop).
        with torch.no_grad():
            if self.mode == 'exp7':
                out = self.model(inputs['input_ids'], inputs['attention_mask'])
                # Exp 7 outputs logits for relations and domains
                rel_logits = out['rel_logits'] 
                dom_logits = out['domain_logits']
                action_indices = None # Exp 7 has no RL actions
            else:
                # Exp 9 outputs action_logits (TIGHT/MEDIUM/LOOSE/STOP)
                action_logits, _, rel_logits, dom_logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                action_probs = F.softmax(action_logits, dim=-1)
                action_indices = torch.argmax(action_probs, dim=-1)[0] # Shape [4]

        # 3. DOMAIN PREDICTION
        # The model first predicts which "category" (Domain) the question belongs to.
        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]
        print(f"[DATA] Predicted Domain: {domain_name}")

        # 4. HOP-BY-HOP REASONING
        current_entities = {topic_mid}
        final_path = []

        for h in range(4): # Max 4 hops
            print(f"\n--- HOP {h+1} ---")
            
            # Action Selection (Exp 9 Only)
            if action_indices is not None:
                action = action_indices[h].item()
                action_map = {0: "TIGHT", 1: "MEDIUM", 2: "LOOSE", 3: "STOP"}
                print(f"[RL] Agent chose Action: {action_map[action]}")
                if action == 3: # STOP
                    break
            else:
                # Exp 7 Default: Check the stop probability
                # (Logic usually handled by a stop_logits layer, simplified here)
                action = 0 # Assume TIGHT for Exp 7 baseline

            # Relation Selection
            # We look at the relation logits for this specific hop 'h'
            hop_rel_logits = rel_logits[0, h]
            
            if action == 0: # TIGHT: Top-1 Relation only
                top_rel_id = torch.argmax(hop_rel_logits).item()
                selected_rels = [self.id2rel[top_rel_id]]
            
            elif action == 1: # MEDIUM: Top-5 Relations
                top5_ids = torch.topk(hop_rel_logits, 5).indices.tolist()
                selected_rels = [self.id2rel[rid] for rid in top5_ids]
            
            elif action == 2: # LOOSE: Domain Fallback
                # Find all relations that belong to the predicted domain
                # (Implementation depends on having a rel-to-domain mapping)
                selected_rels = [r for r in self.id2rel.values() if domain_name in r]
            
            print(f"[PLAN] Selected Relations to explore: {selected_rels[:3]}...")

            # KG TRAVERSAL (The "Execution" Phase)
            # We move from our current entities to the next ones using the KG
            next_entities = set()
            for ent in current_entities:
                for rel_name in selected_rels:
                    # Forward traversal
                    for r, tgt in self.kg['forward'].get(ent, []):
                        if r == rel_name: next_entities.add(tgt)
                    # Backward traversal
                    for r, src in self.kg['backward'].get(ent, []):
                        if r == rel_name: next_entities.add(src)
            
            if not next_entities:
                print("[KG] No more paths found. Dead end.")
                break
                
            current_entities = next_entities
            final_path.append(selected_rels[0]) # Tracking the primary path
            print(f"[KG] Reached {len(current_entities)} entities.")

        # 5. RESULT FORMATTING
        print("\n[RESULT] Reasoning Complete.")
        print(f"Final Answer Entities: {[self.mid2name.get(e, e) for e in list(current_entities)[:5]]}")

if __name__ == "__main__":
    # Example Execution
    # Question: "What are the films starring the actor from Inception?"
    walker = DetailedInferenceWalkthrough(mode='exp9')
    walker.run("What movies did the actor who played Dom Cobb in Inception star in?", "m.0642vqv")
