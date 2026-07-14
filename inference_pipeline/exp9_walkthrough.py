"""
================================================================================
EXPERIMENT 9: RL META-CONSTRAINT AGENT (STEP-BY-STEP WALKTHROUGH)
================================================================================
Architecture: 
- Backbone: Exp 7 Unified Planner (RoBERTa + Transformer)
- RL Head: Action Policy (TIGHT, MEDIUM, LOOSE, STOP)
- Dynamic Width Search

Unlike Exp 7, this model adaptively decides how "broad" its search should be 
at each hop based on its confidence in the reasoning path.
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

class Exp9Walkthrough:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Initializing Exp 9 (RLMC) Pipeline on {self.device}...")
        
        # 1. LOAD MAPPINGS
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. LOAD KNOWLEDGE GRAPH
        kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
        self.kg = torch.load(kg_path, map_location='cpu')
        
        # 3. LOAD ENTITY NAMES
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

        # 4. INITIALIZE MODEL
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        
        # Exp 9 reuses the Exp 7 backbone but adds a Policy Head
        base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.model = RLConstraintAgent(base_model).to(self.device)
        
        # Load the RL Agent weights
        ckpt = os.path.join(ROOT, 'checkpoints/exp9_rlmc_best.pt')
        print(f"[Init] Loading RL weights from {ckpt}...")
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        else:
            print("[Warning] Checkpoint not found. Pipeline will behave randomly.")

        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def run_inference(self, question, topic_mid):
        """
        Walks through the dynamic reasoning process of the RLMC agent.
        """
        print(f"\n[STEP 1: TEXT PROCESSING]")
        print(f"Input Question: \"{question}\"")
        
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # --- FORWARD PASS ---
        with torch.no_grad():
            # In Exp 9, the model outputs Action Logits along with Relation Logits
            action_logits, _, rel_logits, dom_logits = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # [STEP 2: DOMAIN CONTEXT]
        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]
        print(f"Predicted Domain Context: {domain_name}")

        # [STEP 3: ADAPTIVE PLANNING]
        current_entities = {topic_mid}
        path_taken = []
        
        # Action Map for readability
        ACTIONS = {0: "TIGHT (Top-1)", 1: "MEDIUM (Top-5)", 2: "LOOSE (Domain-Wide)", 3: "STOP"}

        for h in range(4):
            print(f"\n--- Reasoning Hop {h+1} ---")
            
            # RL Policy: Decide the width of exploration
            # This is the core of Experiment 9
            hop_action_probs = F.softmax(action_logits[0, h], dim=-1)
            selected_action = torch.argmax(hop_action_probs).item()
            print(f"  RL Policy Decision: {ACTIONS[selected_action]}")

            if selected_action == 3: # STOP action
                print("  -> Agent decided to finalize the answer here.")
                break

            # Get the predicted relations from the backbone
            hop_rel_logits = rel_logits[0, h]
            
            # --- META-CONSTRAINT LOGIC ---
            if selected_action == 0: # TIGHT
                # Only explore the single most likely relation
                top1_id = torch.argmax(hop_rel_logits).item()
                active_rels = [self.id2rel[top1_id]]
            
            elif selected_action == 1: # MEDIUM
                # Explore the top 5 relations to handle ambiguity
                top5_ids = torch.topk(hop_rel_logits, 5).indices.tolist()
                active_rels = [self.id2rel[rid] for rid in top5_ids]
            
            elif selected_action == 2: # LOOSE
                # Extreme fallback: Explore all relations that match the predicted domain
                active_rels = [r for r in self.id2rel.values() if domain_name in r]
            
            print(f"  Exploring {len(active_rels)} relation paths...")

            # [STEP 4: KG EXECUTION]
            next_entities = set()
            for ent in current_entities:
                for rel_name in active_rels:
                    # Look up neighbors in the Knowledge Graph
                    for r, tgt in self.kg['forward'].get(ent, []):
                        if r == rel_name: next_entities.add(tgt)
                    for r, src in self.kg['backward'].get(ent, []):
                        if r == rel_name: next_entities.add(src)
            
            if not next_entities:
                print(f"  !! Reasoning failed at this hop. No triples found.")
                break
                
            current_entities = next_entities
            path_taken.append(active_rels[0] if len(active_rels)==1 else "MULTI_PATH")
            print(f"  -> Reached {len(current_entities)} entities.")

        print("\n[STEP 5: FINAL RESULT]")
        final_names = [self.mid2name.get(e, e) for e in list(current_entities)[:10]]
        print(f"Entities Found: {final_names}")

if __name__ == "__main__":
    walker = Exp9Walkthrough()
    # Question with potential ambiguity (multiple possible relations)
    walker.run_inference("Where did the 'Country Nation World Tour' concert artist go to college", "m.010qhfmm")
