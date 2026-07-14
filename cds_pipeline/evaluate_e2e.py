"""
evaluate_e2e.py — End-to-End Pipeline Evaluation
================================================

Evaluates a given Phase 1 Agent (e.g., Exp9 RLMC or Exp15 STRL) paired with 
the fixed CDS Pipeline (Phase 2 & 3).

Unlike `evaluate.py` which only ranks pre-harvested candidate sets, this script:
1. Takes a natural language question + topic entity
2. Uses the Agent to perform multi-hop graph traversal and build the candidate beam.
3. Uses the CDS Pipeline to rank the candidate beam.

Usage:
  python -m cds_pipeline.evaluate_e2e --agent exp9
  python -m cds_pipeline.evaluate_e2e --agent exp15
  python -m cds_pipeline.evaluate_e2e --agent ensemble
"""
import os, sys, json, torch, lmdb, pickle
from transformers import RobertaTokenizer
from tqdm import tqdm
from argparse import ArgumentParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cds_pipeline.pipeline import CDSPipeline
from utils.sparql_parser import find_reasoning_path
from inference_pipeline.model import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from train.exp15_strl import STRLAgent, RelationEmbeddingBank

class EndToEndEvaluator:
    def __init__(self, agent_type="exp9", s3_version="v2", s2_version="v1", bypass_stage1=False, bypass_stage2=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[E2E] Initializing {agent_type.upper()} Agent + CDS S2={s2_version} S3={s3_version} on {self.device}")
        
        self.agent_type = agent_type
        self.bypass_stage1 = bypass_stage1
        self.bypass_stage2 = bypass_stage2
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Load KG Environment
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
        
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')
        self.id2dom = {v: k for k, v in self.dom2id.items()}
        
        # Load Agent(s)
        base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        
        # exp15 agent (always needed for ensemble)
        if agent_type in ("exp15", "ensemble"):
            self.agent_exp15 = STRLAgent(base).to(self.device)
            ckpt15 = os.path.join(ROOT, 'checkpoints/exp15_strl_best.pt')
            self.agent_exp15.load_state_dict(torch.load(ckpt15, map_location=self.device))
            self.agent_exp15.eval()
            self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)

        # exp9 agent (always needed for ensemble)
        if agent_type in ("exp9", "ensemble"):
            base9 = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
            ckpt9_base = os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt')
            base9.load_state_dict(torch.load(ckpt9_base, map_location=self.device), strict=False)
            self.agent_exp9 = RLConstraintAgent(base9).to(self.device)
            
            ckpt9_rlmc = os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt')
            if os.path.exists(ckpt9_rlmc):
                self.agent_exp9.load_state_dict(torch.load(ckpt9_rlmc, map_location=self.device))
            
            self.agent_exp9.eval()

        # Convenience alias for single-agent modes
        if agent_type == "exp15":
            self.agent = self.agent_exp15
        elif agent_type == "exp9":
            self.agent = self.agent_exp9
            
        # Load CDS Pipeline
        self.cds = CDSPipeline(device=self.device, s3_version=s3_version,
                               s2_version=s2_version, bypass_stage1=bypass_stage1,
                               bypass_stage2=bypass_stage2)
        
    def kg_lookup(self, entities, rels):
        next_entities = set()
        rels_set = set(rels)
        with self.env.begin() as txn:
            for ent in entities:
                f_data = txn.get(f"f:{ent}".encode('utf-8'))
                if f_data:
                    for r, tgt in pickle.loads(f_data):
                        if r in rels_set:
                            next_entities.add(tgt)
                b_data = txn.get(f"b:{ent}".encode('utf-8'))
                if b_data:
                    for r, src in pickle.loads(b_data):
                        if r in rels_set:
                            next_entities.add(src)
        return next_entities

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1 traversal helpers — return a dict:  path_tuple -> set(mid)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _traverse_exp15(self, inputs, topic_entity):
        """
        Exp 21 Beam Search for Exp15 (STRL).

        Instead of collapsing every hop into a single flat entity set, we
        maintain a dictionary  paths: tuple[str] -> (set[mid], score) where each key
        is the sequence of top-1 relation names chosen so far.

        At each hop we expand every active beam with the top-K relations
        (K determined by the action head), compute the accumulated path score,
        and prune to the top 15 paths to prevent exponential path explosion.
        """
        fwd = self.agent_exp15(inputs['input_ids'], inputs['attention_mask'])

        # paths: path_tuple -> (set of entity MIDs, score)
        paths = {(): ({topic_entity}, 0.0)}

        for h in range(4):
            action = torch.argmax(fwd['action_logits'][0, h]).item()
            if action == 3:
                break  # STOP action — all beams terminate here

            sims = torch.mv(self.rel_emb_bank.all(), fwd['hop_reprs'][0, h])
            k = {0: 5, 1: 10, 2: 50}.get(action, 5)
            
            top_k_indices = torch.topk(sims, k).indices.tolist()
            top_k_rels = [self.id2rel[rid] for rid in top_k_indices]
            rel_scores = {self.id2rel[rid]: sims[rid].item() for rid in top_k_indices}

            new_paths = {}
            for path_tuple, (entities, path_score) in paths.items():
                for rel in top_k_rels:
                    next_ents = self.kg_lookup(entities, [rel])
                    if not next_ents:
                        continue
                    if len(next_ents) > 1000:
                        import random
                        next_ents = set(random.sample(list(next_ents), 1000))
                    new_key = path_tuple + (rel,)
                    new_score = path_score + rel_scores[rel]
                    
                    if new_key in new_paths:
                        existing_ents, existing_score = new_paths[new_key]
                        new_paths[new_key] = (existing_ents | next_ents, max(existing_score, new_score))
                    else:
                        new_paths[new_key] = (next_ents, new_score)

            if not new_paths:
                break

            # Prune to top 15 paths by score
            if len(new_paths) > 15:
                sorted_paths = sorted(new_paths.items(), key=lambda x: x[1][1], reverse=True)
                paths = dict(sorted_paths[:15])
            else:
                paths = new_paths

        return {k: v[0] for k, v in paths.items()}

    @torch.no_grad()
    def _traverse_exp9(self, inputs, topic_entity):
        """
        Exp 21 Beam Search for Exp9 (RLMC).

        Exp9 selects a single action+relation per hop, but when LOOSE/MEDIUM actions
        are taken it can expand paths. We track scores and prune to top 15 paths
        at each hop to avoid explosion.
        """
        action_logits, _, rel_logits, dom_logits = self.agent_exp9(
            inputs['input_ids'], inputs['attention_mask']
        )
        dom_name = self.id2dom[torch.argmax(dom_logits, dim=-1).item()]

        # paths: path_tuple -> (set of entity MIDs, score)
        paths = {(): ({topic_entity}, 0.0)}

        for h in range(4):
            action = torch.argmax(action_logits[0, h]).item()
            if action == 3:
                break
            
            if action == 0:
                top_k_indices = [torch.argmax(rel_logits[0, h]).item()]
            elif action == 1:
                top_k_indices = torch.topk(rel_logits[0, h], 5).indices.tolist()
            else:
                # LOOSE (domain fallback)
                domain_rels = [(rid, rel_logits[0, h, rid].item()) 
                               for rid, r in self.id2rel.items() if dom_name in r]
                domain_rels.sort(key=lambda x: x[1], reverse=True)
                top_k_indices = [x[0] for x in domain_rels[:50]]

            top_k_rels = [self.id2rel[rid] for rid in top_k_indices]
            rel_scores = {self.id2rel[rid]: rel_logits[0, h, rid].item() for rid in top_k_indices}

            new_paths = {}
            for path_tuple, (entities, path_score) in paths.items():
                for rel in top_k_rels:
                    next_ents = self.kg_lookup(entities, [rel])
                    if not next_ents:
                        continue
                    if len(next_ents) > 1000:
                        import random
                        next_ents = set(random.sample(list(next_ents), 1000))
                    new_key = path_tuple + (rel,)
                    new_score = path_score + rel_scores[rel]
                    
                    if new_key in new_paths:
                        existing_ents, existing_score = new_paths[new_key]
                        new_paths[new_key] = (existing_ents | next_ents, max(existing_score, new_score))
                    else:
                        new_paths[new_key] = (next_ents, new_score)

            if not new_paths:
                break

            # Prune to top 15 paths by score
            if len(new_paths) > 15:
                sorted_paths = sorted(new_paths.items(), key=lambda x: x[1][1], reverse=True)
                paths = dict(sorted_paths[:15])
            else:
                paths = new_paths

        return {k: v[0] for k, v in paths.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Flatten beam paths into a candidate list with per-candidate path strings
    # ──────────────────────────────────────────────────────────────────────────

    def _paths_to_candidates(self, paths, gold_mids):
        """
        Flatten a  path_tuple -> set(mid)  dict into a candidate list.

        Each candidate dict carries its own `path` field (the specific path
        string that produced it), which Stage 2 and Stage 3 will use for
        path-aware scoring.

        If the same MID appears in multiple beams we keep the entry whose
        path was generated first (shortest path_tuple).
        """
        seen = {}  # mid -> candidate dict (dedup, keep first/shortest)
        for path_tuple, entities in sorted(paths.items(), key=lambda x: len(x[0])):
            path_str = " -> ".join(path_tuple)
            for mid in entities:
                if mid not in seen:
                    seen[mid] = {
                        "mid": mid,
                        "name": self.mid2name.get(mid, "Unknown"),
                        "is_gold": mid in gold_mids,
                        "path": path_str,
                    }
        return list(seen.values())

    # ──────────────────────────────────────────────────────────────────────────
    # Main evaluation loop
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, samples, max_samples=None):
        if max_samples:
            samples = samples[:max_samples]
            
        stats = {'hit1': 0, 'hit_s2': 0, 'hit_s1': 0, 'hit_n': 0, 'count': 0}
        
        for s in tqdm(samples, desc=f"E2E: {self.agent_type} + CDS"):
            q = s['q']
            inputs = self.tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # ─── PHASE 1: AGENT TRAVERSAL (Exp 21 Beam Search) ────────────────
            if self.agent_type == 'exp15':
                paths = self._traverse_exp15(inputs, s['te'])

            elif self.agent_type == 'exp9':
                paths = self._traverse_exp9(inputs, s['te'])

            elif self.agent_type == 'ensemble':
                # Exp 22: merge beams from both agents; deduplicate by MID
                paths15 = self._traverse_exp15(inputs, s['te'])
                paths9  = self._traverse_exp9(inputs, s['te'])
                # Merge: union entity sets for matching path_tuples, else keep both
                paths = dict(paths15)
                for path_tuple, entities in paths9.items():
                    if path_tuple in paths:
                        paths[path_tuple] |= entities
                    else:
                        paths[path_tuple] = entities

            # ─── EVAL PHASE 1 ─────────────────────────────────────────────────
            all_found_mids = set()
            for ents in paths.values():
                all_found_mids |= ents

            stats['count'] += 1
            if any(mid in s['gold'] for mid in all_found_mids):
                stats['hit_n'] += 1
                
                # ─── Build per-candidate-path candidate list ───────────────────
                candidates = self._paths_to_candidates(paths, s['gold'])

                # Cap extremely large candidate sets to prevent OOM / hanging
                if len(candidates) > 5000:
                    import random
                    golds = [c for c in candidates if c['is_gold']]
                    negs  = [c for c in candidates if not c['is_gold']]
                    candidates = golds + random.sample(negs, min(5000, len(negs)))

                # ─── PHASE 2: CDS RANKING ──────────────────────────────────────
                # Pass None as path — pipeline will use per-candidate c['path']
                ranked, cands_s2, cands_s3 = self.cds.rank(
                    q, candidates, path=None, return_intermediates=True
                )
                
                if cands_s2 and any(c['is_gold'] for c in cands_s2):
                    stats['hit_s1'] += 1
                if cands_s3 and any(c['is_gold'] for c in cands_s3):
                    stats['hit_s2'] += 1
                if ranked and ranked[0]['is_gold']:
                    stats['hit1'] += 1

        print("\n" + "="*50)
        print(f" END-TO-END RESULTS: {self.agent_type.upper()} + CDS")
        print("="*50)
        print(f" Total Questions  : {stats['count']}")
        print(f" Reasoning Recall : {stats['hit_n']/stats['count']*100:.2f}% (Input to Stage 1)")
        print(f" Hit@200 (Stage 1): {stats['hit_s1']/stats['count']*100:.2f}% (Output of MiniLM Bi-Encoder)")
        print(f" Hit@50  (Stage 2): {stats['hit_s2']/stats['count']*100:.2f}% (Output of Path-Aware MPNet)")
        print(f" Final Hit@1 (S3) : {stats['hit1']/stats['count']*100:.2f}% (Output of BGE Cross-Encoder)")
        print("="*50)

def main():
    parser = ArgumentParser()
    parser.add_argument("--agent", default="exp9", choices=["exp9", "exp15", "ensemble"])
    parser.add_argument("--s3", type=str, choices=["v2", "v3", "v4", "v5", "v6", "v7", "v8_gen", "v9_rl_policy", "v10_pure_rl", "v11_gen_sc", "v12_t5_mc", "v13_t5_cot", "v14_t5_pointer", "v15_t5_listwise", "v16_bge_cross", "v17_bge_infonce", "v18_t5_dpo"], default="v6")
    parser.add_argument("--s2", default="v1",
                        choices=["v1", "v2"],
                        help="Stage 2 version: v1=SoftMargin (default), v2=KL-Listwise (Exp 25)")
    parser.add_argument("--bypass_stage1", action="store_true",
                        help="Skip the Stage 1 generic bi-encoder bottleneck and feed all candidates to Stage 2.")
    parser.add_argument("--bypass_stage2", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    print("[E2E] Loading Dev Set...")
    with open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    samples = []
    for item in data:
        path = find_reasoning_path(item['sparql'])
        if not path: continue
        samples.append({
            'q': item['question'], 
            'te': path[0][0].replace("ns:", ""), 
            'gold': set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
        })
        
    print(f"[E2E] Prepared {len(samples)} valid questions from Dev Set")
    
    evaluator = EndToEndEvaluator(agent_type=args.agent, s3_version=args.s3,
                                   s2_version=args.s2, bypass_stage1=args.bypass_stage1,
                                   bypass_stage2=args.bypass_stage2)
    evaluator.evaluate(samples, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
