import os, sys, json, torch, time, lmdb, pickle
import torch.nn.functional as F
from transformers import RobertaTokenizer

# Ensure we can import from the root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner

class Exp7Optimized:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- OPTIMIZED LOADING PART ---
        print(f"[Init] Starting High-Speed Pipeline (LMDB Mode)...")
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
        self.model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        
        checkpoint_path = os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt')
        print(f"[Init] Loading weights from {checkpoint_path}...")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        print("[Init] Pipeline Ready.\n")

    def run_inference(self, question, topic_mid):
        """
        Walks through the inference process with detailed printing (Restored from Walkthrough).
        """
        print(f"\n[STEP 1: TEXT PROCESSING]")
        print(f"Input Question: \"{question}\"")
        
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # --- FORWARD PASS ---
        with torch.no_grad():
            out = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # [STEP 2: DOMAIN PREDICTION]
        dom_logits = out['domain_logits']
        pred_dom_id = torch.argmax(dom_logits, dim=-1).item()
        domain_name = self.id2dom[pred_dom_id]
        print(f"Predicted Domain: {domain_name}")

        # [STEP 3: MULTI-HOP PLANNING]
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
            
            # --- OPTIMIZED KG EXECUTION (LMDB LOOKUP) ---
            next_entities = set()
            with self.env.begin() as txn:
                for ent in current_entities:
                    # Forward Edges
                    f_data = txn.get(f"f:{ent}".encode('utf-8'))
                    if f_data:
                        neighbors = pickle.loads(f_data)
                        for r, tgt in neighbors:
                            if r == relation_name: next_entities.add(tgt)
                    
                    # Backward Edges
                    b_data = txn.get(f"b:{ent}".encode('utf-8'))
                    if b_data:
                        neighbors = pickle.loads(b_data)
                        for r, src in neighbors:
                            if r == relation_name: next_entities.add(src)
            # --------------------------------------------

            if not next_entities:
                print(f"  !! Found no paths for '{relation_name}' in KG. Stopping.")
                break
                
            current_entities = next_entities
            path_taken.append(relation_name)
            
            # Detailed sample output as in original
            names = [self.mid2name.get(e, e) for e in list(current_entities)[:3]]
            print(f"  -> Found {len(current_entities)} entities: {names}...")

        print("\n[STEP 5: FINAL ANSWER]")
        print(f"Path: {' -> '.join(path_taken)}")
        final_answers = [self.mid2name.get(e, e) for e in list(current_entities)[:10]]
        print(f"Answer List: {final_answers}")

if __name__ == "__main__":
    app = Exp7Optimized()
    # Test Question
    app.run_inference("Lou Seal is the mascot for the team that last won the World Series when", "m.03_dwn")
