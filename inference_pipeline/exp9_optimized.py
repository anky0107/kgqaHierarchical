import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer

# Ensure we can import from the root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent

class Exp9Optimized:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Initializing Exp 9 (RLMC) Optimized...")
        
        # --- OPTIMIZED LOADING PART ---
        start_load = time.time()
        
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # Instant KG access using LMDB
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        load_duration = time.time() - start_load
        print(f"[METRIC] KG Loading (LMDB) took: {load_duration:.4f} seconds")
        # ------------------------------

        # Load Weights & Model (Restored from Original Walkthrough)
        print("[Init] Loading Entity Names...")
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.model = RLConstraintAgent(base_model).to(self.device)
        
        # Use epoch_9 as the 'best' candidate (restored from your checkpoint list)
        checkpoint_path = os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt')
        print(f"[Init] Loading weights from {checkpoint_path}...")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        print("[Init] Pipeline Ready.\n")

    def run_inference(self, question, topic_mid):
        """
        Full Experiment 9 Reasoning Logic with Step-by-Step Printing.
        """
        print(f"\n[STEP 1: TEXT PROCESSING]")
        print(f"Input Question: \"{question}\"")
        
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            # In Exp 9, the model outputs Action Logits along with Relation Logits
            action_logits, _, rel_logits, dom_logits = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # [STEP 2: DOMAIN CONTEXT]
        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]
        print(f"Predicted Domain Context: {domain_name}")

        # [STEP 3: ADAPTIVE PLANNING]
        current_entities = {topic_mid}
        ACTIONS = {0: "TIGHT (Top-1)", 1: "MEDIUM (Top-5)", 2: "LOOSE (Domain-Wide)", 3: "STOP"}

        for h in range(4):
            print(f"\n--- Reasoning Hop {h+1} ---")
            
            # RL Policy: Decide the width of exploration
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
                top1_id = torch.argmax(hop_rel_logits).item()
                active_rels = [self.id2rel[top1_id]]
            elif selected_action == 1: # MEDIUM
                top5_ids = torch.topk(hop_rel_logits, 5).indices.tolist()
                active_rels = [self.id2rel[rid] for rid in top5_ids]
            elif selected_action == 2: # LOOSE
                active_rels = [r for r in self.id2rel.values() if domain_name in r]
            
            print(f"  Exploring {len(active_rels)} relation paths...")

            # [STEP 4: OPTIMIZED KG EXECUTION (LMDB LOOKUP)]
            next_entities = set()
            with self.env.begin() as txn:
                for ent in current_entities:
                    for rel_name in active_rels:
                        # Forward
                        f_data = txn.get(f"f:{ent}".encode('utf-8'))
                        if f_data:
                            for r, tgt in pickle.loads(f_data):
                                if r == rel_name: next_entities.add(tgt)
                        # Backward
                        b_data = txn.get(f"b:{ent}".encode('utf-8'))
                        if b_data:
                            for r, src in pickle.loads(b_data):
                                if r == rel_name: next_entities.add(src)
            # --------------------------------------------

            if not next_entities:
                print(f"  !! Reasoning failed at this hop. No triples found.")
                break
                
            current_entities = next_entities
            names = [self.mid2name.get(e, e) for e in list(current_entities)[:3]]
            print(f"  -> Reached {len(current_entities)} entities: {names}...")

        print("\n[STEP 5: FINAL RESULT]")
        final_names = [self.mid2name.get(e, e) for e in list(current_entities)[:10]]
        print(f"Final Answers: {final_names}")

if __name__ == "__main__":
    app = Exp9Optimized()
    # Test Question
    app.run_inference("Lou Seal is the mascot for the team that last won the World Series when", "m.03_dwn")
