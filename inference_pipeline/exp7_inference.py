import os, sys, json, torch
import torch.nn.functional as F
from transformers import RobertaTokenizer

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.model import ScaledUnifiedPlanner

class Exp7InferencePipeline:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        
        # 1. Load Mappings
        print("[Init] Loading vocabulary and mappings...")
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. Load KG
        print("[Init] Loading Knowledge Graph...")
        kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
        self.kg = torch.load(kg_path, map_location='cpu') # dict with 'forward' and 'backward'
        
        # 3. Load Entity Names
        print("[Init] Loading Entity Names...")
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        self.name2mid = {v.lower(): k for k, v in self.mid2name.items()}

        # 4. Initialize Model
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        self.model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt')
        
        print(f"[Init] Loading weights from {checkpoint_path}...")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        print("[Init] Pipeline ready.\n")

    def find_entity_mid(self, name):
        """Helper to find MID for a given name."""
        name_lower = name.lower()
        if name_lower in self.name2mid:
            #print("name_lower:" , name_lower)
            #print(name2mid[name_lower])
            return self.name2mid[name_lower]
        # Partial match
        for k, v in self.mid2name.items():
            if name_lower in v.lower():
                return k
        return None

    def get_entity_name(self, mid):
        return self.mid2name.get(mid, mid)

    @torch.no_grad()
    def run_inference(self, question, topic_entity_mid):
        print(f"--- Inference Execution ---")
        print(f"Question: '{question}'")
        print(f"Topic Entity: {self.get_entity_name(topic_entity_mid)} ({topic_entity_mid})")
        print("-" * 30)

        # Step 1: Encoding
        print("[Step 1] Encoding question with RoBERTa-Large...")
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        out = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        hq = out['h_q']
        print(f"  -> Generated question embedding h_q: {hq.shape} (Norm: {torch.norm(hq).item():.4f})")

        # Step 2: Progressive Constraints (Domain & Confidence)
        print("\n[Step 2] Predicting Domain and Confidence...")
        dom_probs = F.softmax(out['domain_logits'], dim=-1)
        conf = out['confidence'].item()
        
        top_dom_id = torch.argmax(dom_probs, dim=-1).item()
        top_dom_name = self.id2dom[top_dom_id]
        top_dom_score = dom_probs[0, top_dom_id].item()
        
        print(f"  -> Predicted Domain: {top_dom_name} (Score: {top_dom_score:.4f})")
        print(f"  -> Model Confidence: {conf:.4f}")

        # Step 3 & 4: Relation Prediction & Path Execution
        print("\n[Step 3 & 4] Sequential Path Reasoning & KG Traversal...")
        
        current_entities = {topic_entity_mid}
        execution_path = []
        
        for h in range(self.model.max_hops):
            print(f"\n--- Hop {h+1} ---")
            
            # Relation Prediction for this hop
            rel_logits = out['rel_logits'][0, h]
            stop_logit = out['stop_logits'][0, h]
            stop_prob = torch.sigmoid(stop_logit).item()
            
            rel_probs = F.softmax(rel_logits, dim=-1)
            top_rel_id = torch.argmax(rel_probs).item()
            top_rel_name = self.id2rel[top_rel_id]
            top_rel_score = rel_probs[top_rel_id].item()
            
            print(f"  Predicted Relation: {top_rel_name} (Score: {top_rel_score:.4f})")
            print(f"  Keep-Going Prob:    {stop_prob:.4f}")
            
            if stop_prob < 0.5:
                print(f"  -> [STOP] Model suggested stopping (Keep-Going Prob: {stop_prob:.4f}). Ending traversal.")
                break
            
            # Execution
            next_entities = set()
            print(f"  Traversing KG from {len(current_entities)} entities...")
            
            for ent in current_entities:
                # Forward
                for rel, tgt in self.kg['forward'].get(ent, []):
                    if rel == top_rel_name:
                        next_entities.add(tgt)
                # Backward
                for rel, tgt in self.kg['backward'].get(ent, []):
                    if rel == top_rel_name:
                        next_entities.add(tgt)
            
            if not next_entities:
                print(f"  !! No transitions found in KG for relation '{top_rel_name}'. Stopping traversal.")
                break
            
            current_entities = next_entities
            execution_path.append(top_rel_name)
            print(f"  -> Reached {len(current_entities)} entities.")
            if len(current_entities) <= 5:
                print(f"     Samples: {[self.get_entity_name(e) for e in list(current_entities)[:5]]}")
            else:
                print(f"     Samples: {[self.get_entity_name(e) for e in list(current_entities)[:5]]} ...")

        print("\n" + "=" * 30)
        print("Final Result:")
        print(f"Predicted Path: {' -> '.join(execution_path)}")
        print(f"Answer Entities ({len(current_entities)}):")
        for ent in list(current_entities):
            print(f"  - {self.get_entity_name(ent)} ({ent})")
        
        return {

            'domain': top_dom_name,
            'confidence': conf,
            'path': execution_path,
            'answers': list(current_entities)
        }

if __name__ == "__main__":
    # Minimal Example Usage
    pipeline = Exp7InferencePipeline()
    
    # Example question: "What is the mascot of the team that has Nicholas S. Zeppos as its leader?"
    # Topic Entity: Nicholas S. Zeppos (m.02vymvp)
    
    question = "Where did the 'Country Nation World Tour' concert artist go to college"
    mid = "m.010qhfmm" # Lou Seal (San Francisco Giants Mascot)
    
    pipeline.run_inference(question, mid)


## exec1 
# gold answer present in exploration

##exec2
# what is losses i can go with 
# h r t 
#question + hrt + predicted answers   ///  gold answers ? what can be list of losses

##exec3
# vanilla transformer for single entity prediction 


## exp1 
# rl supervised training