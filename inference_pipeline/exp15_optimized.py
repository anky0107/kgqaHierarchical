"""
Exp 15: High-Speed Optimized Inference Pipeline (STRL-KGQA)
==========================================================

Architecture:
  - Stage 1 & 2: STRLAgent (RoBERTa-Large + PPO Policy)
  - Stage 3: MiniLM Answer Selector (Cross-Encoder)
  - Optimizer: LMDB Memory-Mapped KG

Performance:
  - Instant KG lookups (<1ms)
  - Semantic filtering for opaque bridge entities (e.g. 'Country Nation World Tour')
  - Ranked final answers instead of random sets
"""

import os, sys, json, torch, time, lmdb, pickle
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, AutoTokenizer, AutoModelForSequenceClassification

# Ensure we can import from the root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.exp15_strl import STRLAgent, RelationEmbeddingBank, BEAM_SIZES
from train.exp7_roberta import ScaledUnifiedPlanner

class Exp15Optimized:
    def __init__(self, strl_ckpt=None, selector_ckpt=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Starting STRL Optimized Pipeline (Device: {self.device})")
        start_time = time.time()

        # 1. Load Mappings
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # 2. Load LMDB KG
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        if not os.path.exists(lmdb_path):
            # Fallback to in-memory if LMDB not generated
            print("[Warning] LMDB KG not found. Loading augmented_kg.pt into RAM (this may be slow)...")
            kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
            self.kg_ram = torch.load(kg_path, map_location='cpu')
            self.use_lmdb = False
        else:
            self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            self.use_lmdb = True
            print("[Init] KG loaded via LMDB.")

        # 3. Load Entity Names
        print("[Init] Loading Entity Names...")
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        # 4. Initialize STRL Agent (Stages 1 & 2)
        print("[Init] Initializing STRL Agent...")
        num_dom = len(self.dom2id)
        num_rel = len(self.rel2id)
        base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(self.device)
        self.agent = STRLAgent(base_model).to(self.device)
        
        if strl_ckpt is None:
            strl_ckpt = os.path.join(ROOT, 'checkpoints/exp15_strl_best.pt')
        
        if os.path.exists(strl_ckpt):
            print(f"[Init] Loading STRL weights from {strl_ckpt}...")
            self.agent.load_state_dict(torch.load(strl_ckpt, map_location=self.device))
        else:
            print(f"[Warning] STRL checkpoint not found at {strl_ckpt}. Using random/Exp7 weights.")

        self.agent.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        # 5. Initialize Semantic Teacher Bank (for beam filtering)
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
        self.rel_emb_bank.eval()

        # 6. Initialize Stage 3: Answer Selector
        print("[Init] Initializing Stage 3 Answer Selector (MiniLM)...")
        selector_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.selector_tokenizer = AutoTokenizer.from_pretrained(selector_model_name)
        self.selector_model = AutoModelForSequenceClassification.from_pretrained(selector_model_name, num_labels=1).to(self.device)
        
        if selector_ckpt is None:
            selector_ckpt = os.path.join(ROOT, 'checkpoints/exp15_answer_selector.pt')
        
        if os.path.exists(selector_ckpt):
            print(f"[Init] Loading Answer Selector weights from {selector_ckpt}...")
            self.selector_model.load_state_dict(torch.load(selector_ckpt, map_location=self.device))
        
        self.selector_model.eval()

        print(f"[Init] Pipeline Ready. Load time: {time.time() - start_time:.2f}s\n")

    def kg_get_neighbors(self, mid):
        """Helper to get neighbors from LMDB or RAM."""
        neighbors = []
        if self.use_lmdb:
            with self.env.begin() as txn:
                # Forward
                f_data = txn.get(f"f:{mid}".encode('utf-8'))
                if f_data: neighbors.extend(pickle.loads(f_data))
                # Backward
                b_data = txn.get(f"b:{mid}".encode('utf-8'))
                if b_data: neighbors.extend(pickle.loads(b_data))
        else:
            neighbors.extend(self.kg_ram.get('forward', {}).get(mid, []))
            neighbors.extend(self.kg_ram.get('backward', {}).get(mid, []))
        return neighbors

    def get_semantic_beam_with_filter(self, hop_repr, current_entities, action):
        """
        Filters the semantic beam to only include relations actually reachable in the KG.
        If no reachable relations exist in top-k, falls back to re-ranking all reachable.
        """
        all_embs = self.rel_emb_bank.all()
        sims     = torch.mv(all_embs, hop_repr)
        k        = BEAM_SIZES.get(action, 5)
        
        # Get reachable rels
        reachable_rels = set()
        for mid in current_entities:
            for rel, _ in self.kg_get_neighbors(mid):
                if rel in self.rel2id:
                    reachable_rels.add(self.rel2id[rel])

        # Step 1: Semantic top-k
        top_k = torch.topk(sims, k)
        sem_ids = top_k.indices.tolist()
        
        # Step 2: Intersection
        valid_beam = [rid for rid in sem_ids if rid in reachable_rels]
        
        if not valid_beam and reachable_rels:
            # Fallback: re-rank all reachable relations by semantic score
            fallback = [(rid, sims[rid].item()) for rid in reachable_rels]
            fallback.sort(key=lambda x: x[1], reverse=True)
            valid_beam = [rid for rid, _ in fallback[:k]]
        
        return valid_beam or sem_ids[:k]

    @torch.no_grad()
    def run_inference(self, question, topic_mid):
        print(f"--- STRL Inference ---")
        print(f"Question: \"{question}\"")
        print(f"Starting Entity: {self.mid2name.get(topic_mid, topic_mid)}")
        
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        fwd = self.agent(inputs['input_ids'], inputs['attention_mask'])
        
        current_entities = {topic_mid}
        path_taken = []
        path_confidences = []
        
        for h in range(4): # max 4 hops
            print(f"\n[Hop {h+1}]")
            
            # 1. Get Action from Policy
            action_logits = fwd['action_logits'][0, h]
            action = torch.argmax(action_logits).item()
            action_names = ["TIGHT", "MEDIUM", "LOOSE", "STOP"]
            print(f"  Policy Action: {action_names[action]}")
            
            if action == 3: # STOP
                print("  -> Policy reached STOP condition.")
                break
            
            # 2. Get Semantic Beam with KG Filter
            hop_repr = fwd['hop_reprs'][0, h]
            
            # Track semantic confidence for this hop
            all_embs = self.rel_emb_bank.all()
            sims = torch.mv(all_embs, hop_repr)
            
            beam_ids = self.get_semantic_beam_with_filter(hop_repr, current_entities, action)
            beam_rels = [self.id2rel[rid] for rid in beam_ids]
            
            # Best reachable sim
            best_hop_sim = sims[beam_ids[0]].item()
            path_confidences.append(best_hop_sim)
            print(f"  Grounded Beam: {beam_rels} (Top Sim: {best_hop_sim:.4f})")
            
            # 3. Execute KG Traversal
            next_entities = set()
            beam_rel_set = set(beam_rels)
            for mid in current_entities:
                for rel, tgt in self.kg_get_neighbors(mid):
                    if rel in beam_rel_set:
                        next_entities.add(tgt)
            
            if not next_entities:
                print(f"  !! Traversal Dead-end for {beam_rels}. Stopping.")
                break
                
            current_entities = next_entities
            path_taken.append(beam_rels[0]) # track primary path
            print(f"  -> Reached {len(current_entities)} entities.")

        # Stage 3: Answer Selection
        print(f"\n[Stage 3: Answer Selection]")
        if not current_entities:
            print("  No entities reached. Result: None")
            return []

        # Collect candidate names
        candidates = []
        for mid in current_entities:
            name = self.mid2name.get(mid, None)
            if name: candidates.append((mid, name))
        
        if not candidates:
            print(f"  Found MIDs but no names in master_mid2name. MIDs: {list(current_entities)[:3]}...")
            return list(current_entities)[:5]

        # Limit candidates to score (e.g. 100) to avoid slow cross-encoding
        candidates = candidates[:100]
        
        # Build cross-encoder inputs
        questions = [question] * len(candidates)
        names = [c[1] for c in candidates]
        
        with torch.no_grad():
            enc = self.selector_tokenizer(questions, names, padding=True, truncation=True, return_tensors='pt').to(self.device)
            selector_logits = self.selector_model(**enc).logits.squeeze(-1)
            
        # --- PATH SAFETY VALVE ---
        # Calculate mean path confidence. If it's low (e.g. < 0.2), we penalize the final score.
        path_score = sum(path_confidences) / len(path_confidences) if path_confidences else 0.0
        print(f"  Path Safety Score: {path_score:.4f}")
        
        # Combined score: selector_confidence * path_score
        scores = torch.sigmoid(selector_logits) * path_score
            
        # Rank by score
        ranked_indices = torch.argsort(scores, descending=True)
        results = []
        print("  Top Ranked Answers (Combined Confidence):")
        for i in range(min(5, len(ranked_indices))):
            idx = ranked_indices[i].item()
            final_conf = scores[idx].item()
            mid, name = candidates[idx]
            results.append((name, final_conf))
            print(f"    {i+1}. {name} (Final: {final_conf:.4f} | Selector: {torch.sigmoid(selector_logits[idx]).item():.4f})")
            
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="where did the concert artist go to college")
    parser.add_argument("--topic", type=str, default="m.010qhfmm")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--selector_ckpt", type=str, default=None)
    args = parser.parse_args()

    pipeline = Exp15Optimized(strl_ckpt=args.ckpt, selector_ckpt=args.selector_ckpt)
    pipeline.run_inference(args.question, args.topic)
