import os, sys, json, torch, lmdb, pickle
from tqdm import tqdm
from transformers import RobertaTokenizer

"""
collect_candidates.py — CDS Pipeline: F1 Candidate Collection
==============================================================

Paper Section: §V-C  "Cascading Dual-Stage Filtering (CDS) — Stage I (F1)"

Purpose
-------
This module is the first stage (F1) of the three-stage Cascading Dual-Stage
(CDS) answer-selection pipeline.  It re-uses the already-trained STRL agent
(Exp-15 / §IV) to perform a constrained KG traversal from the topic entity and
collects the **raw candidate entity pool** that will be passed to the downstream
F2 (MPNet path-aware ranker) and F3 (Flan-T5 generative judge) stages.

Pipeline position
-----------------
  SPARQL / CWQ question
        │
        ▼
  [F1 – collect_candidates.py]  ← THIS FILE
        │  produces: exp16_cds_{train,dev}.json
        ▼
  [F2 – train_f2_path_ranker.py]  top-50 retained by MPNet
        │
        ▼
  [F3 – train_f3_sft.py → train_f3_dpo.py]  Flan-T5 final answer

Inputs
------
- CWQ split JSON   : data/cwq_{train,dev}.json
                     Each record must contain 'question', 'sparql', and
                     'answers' (list of {'answer_id': ...}).
- STRL checkpoint  : checkpoints/exp15_strl_epoch_19.pt
- Relation mappings: data/processed_entity/{relation2id,domain2id}.pt
- KG LMDB store    : data/processed_kg/augmented_kg_lmdb
                     (forward 'f:<mid>' and backward 'b:<mid>' adjacency lists)
- MID-to-name map  : data/master_mid2name.json

Outputs
-------
- data/exp16_cds_train.json  (2 000-sample subset used to train F2)
- data/exp16_cds_dev.json    (full dev set for evaluation)

Each output record has the schema:
    {
        "question"   : str,
        "path"       : [[rel_name, ...], ...],   # per-hop relation beam
        "candidates" : [{"mid": str, "name": str, "is_gold": bool}, ...]
    }

Key hyperparameters
-------------------
- Beam widths per action: {0: 5, 1: 10, 2: 50}  (action indices 0/1/2)
- Max traversal depth   : 4 hops
- Action index 3        : STOP signal — terminates traversal early

How it works
------------
1. Parse the gold reasoning path from the SPARQL query via find_reasoning_path.
2. Identify the topic entity (first node in the path).
3. Run the STRL agent on the question to obtain per-hop action logits and
   dense hop-representation vectors.
4. At each hop:
   a. Take the argmax action to determine beam width k.
   b. Compute cosine similarities between the hop representation and ALL
      relation embeddings in the RelationEmbeddingBank.
   c. Intersect the top-k relations with those actually reachable from the
      current entity frontier (pruning hallucinated edges).
   d. Expand the frontier via kg_lookup (bidirectional LMDB traversal).
5. Annotate each final-frontier entity with its human-readable name and
   a is_gold flag (True iff the MID appears in the gold answer set).
"""

# Add root to sys.path so sibling packages (train/, utils/, etc.) are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.exp15_strl import STRLAgent, RelationEmbeddingBank
from inference_pipeline.model import ScaledUnifiedPlanner
from utils.sparql_parser import find_reasoning_path

# ──────────────────────────────────────────────────────────────────────────────
# CDSDataCollector
# ──────────────────────────────────────────────────────────────────────────────

class CDSDataCollector:
    """
    Orchestrates the F1 candidate-collection pass for the CDS pipeline.

    Initialisation loads:
      • The trained STRL agent (ScaledUnifiedPlanner wrapped by STRLAgent)
        which provides per-hop action logits and dense hop representations.
      • The RelationEmbeddingBank, a learnable embedding table over all KG
        relations used for semantic beam search.
      • The RoBERTa tokenizer (roberta-large) used to encode questions for
        the STRL agent.
      • A read-only LMDB handle to the augmented KG adjacency lists.
      • The master MID-to-name lookup dictionary.
    """

    def __init__(self, exp15_ckpt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Collector] Initializing with {self.device}")

        # ── Load relation / domain vocabularies ───────────────────────────────
        data_dir = os.path.join(ROOT, 'data/processed_entity')
        # rel2id: str relation name → integer ID  (used for LMDB key matching)
        self.rel2id = torch.load(os.path.join(data_dir, 'relation2id.pt'), map_location='cpu')
        # id2rel: reverse mapping — integer ID → str relation name
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.dom2id = torch.load(os.path.join(data_dir, 'domain2id.pt'), map_location='cpu')

        # ── Instantiate and restore the STRL agent ────────────────────────────
        # ScaledUnifiedPlanner is the base policy network; STRLAgent wraps it
        # with the Self-Taught Reinforcement Learning training interface.
        base = ScaledUnifiedPlanner(len(self.dom2id), len(self.rel2id)).to(self.device)
        self.agent = STRLAgent(base).to(self.device)
        self.agent.load_state_dict(torch.load(exp15_ckpt, map_location=self.device))
        self.agent.eval()  # inference mode — no gradient tracking

        # ── Relation embedding bank ───────────────────────────────────────────
        # Provides a dense embedding for every KG relation; used for cosine-
        # similarity beam search at each hop.
        self.rel_emb_bank = RelationEmbeddingBank(self.id2rel, self.device).to(self.device)
        self.rel_emb_bank.eval()

        # RoBERTa tokenizer: encodes natural-language questions for the STRL agent
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        # ── KG adjacency store (LMDB, read-only) ─────────────────────────────
        # Keys: "f:<mid>" → forward edges [(rel, target_mid), ...]
        #       "b:<mid>" → backward edges [(rel, source_mid), ...]
        lmdb_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg_lmdb')
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        # Human-readable names for Freebase MIDs (used for F3 prompt construction)
        self.mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))

    # ──────────────────────────────────────────────────────────────────────────
    # KG bidirectional lookup
    # ──────────────────────────────────────────────────────────────────────────

    def kg_lookup(self, entities, rels):
        """
        Expand the current entity frontier by one hop along the given relations.

        Uses bidirectional lookup (both "f:" forward and "b:" backward edges) so
        that the CDS pipeline is direction-agnostic — consistent with how the
        augmented KG was constructed (§III-B).

        Parameters
        ----------
        entities : iterable of str  — current frontier MIDs
        rels     : iterable of str  — relation names to follow

        Returns
        -------
        set of str  — MIDs reachable from `entities` via any relation in `rels`
        """
        next_entities = set()
        with self.env.begin() as txn:
            for ent in entities:
                for rel in rels:
                    # Forward direction: entity is the subject
                    f_data = txn.get(f"f:{ent}".encode('utf-8'))
                    if f_data:
                        for r, tgt in pickle.loads(f_data):
                            if r == rel: next_entities.add(tgt)
                    # Backward direction: entity is the object (reverse edges)
                    b_data = txn.get(f"b:{ent}".encode('utf-8'))
                    if b_data:
                        for r, src in pickle.loads(b_data):
                            if r == rel: next_entities.add(src)
        return next_entities

    # ──────────────────────────────────────────────────────────────────────────
    # Main collection loop
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()  # disable autograd — purely inference
    def collect(self, input_file, output_file, limit=None):
        """
        Run the F1 candidate-collection pass over a CWQ split and write results.

        For each question the method:
          1. Parses the SPARQL to obtain the gold reasoning path and topic entity.
          2. Runs one forward pass of the STRL agent to obtain action logits
             (which determine beam width) and per-hop dense representations.
          3. Performs up to 4 hops of semantically-guided, reachability-pruned
             KG traversal.
          4. Annotates the final entity frontier with names and gold labels.

        Parameters
        ----------
        input_file  : str  — path to CWQ split JSON (train/dev)
        output_file : str  — path to write exp16_cds_*.json
        limit       : int or None  — cap on number of questions (for quick tests)
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Optionally truncate for faster development runs
        if limit: data = data[:limit]

        results = []
        for item in tqdm(data, desc=f"Collecting {os.path.basename(input_file)}"):
            # ── Parse gold path from SPARQL ───────────────────────────────────
            path = find_reasoning_path(item['sparql'])
            if not path: continue  # skip malformed / unsupported SPARQL queries

            q  = item['question']
            # The topic entity is the first node of the gold path; strip "ns:" prefix
            te   = path[0][0].replace("ns:", "")
            # Build gold MID set for is_gold annotation later
            gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))

            # ── STRL agent forward pass ───────────────────────────────────────
            # Tokenise the natural-language question with RoBERTa
            inputs = self.tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(self.device)
            # fwd contains:
            #   action_logits  : [1, 4, 4]  — per-hop action distribution
            #   hop_reprs      : [1, 4, H]  — dense query-conditioned hop vectors
            fwd = self.agent(inputs['input_ids'], inputs['attention_mask'])

            # ── Iterative KG traversal ────────────────────────────────────────
            current       = {te}   # entity frontier, initialised to topic entity
            execution_log = []     # records selected relations per hop (for F3 context)

            for h in range(4):  # maximum 4-hop paths in CWQ
                # Determine action: 0 → narrow beam (k=5), 1 → medium (k=10),
                #                   2 → wide beam (k=50),  3 → STOP traversal
                action = torch.argmax(fwd['action_logits'][0, h]).item()
                if action == 3: break  # agent signals end-of-path

                # Dense representation for this hop — used as a "query" vector
                hop_repr = fwd['hop_reprs'][0, h]

                # ── Semantic relation beam ─────────────────────────────────────
                # Compute cosine similarities between the hop query and ALL
                # relation embeddings in one batched matrix-vector product.
                all_embs = self.rel_emb_bank.all()          # [R, H]
                sims     = torch.mv(all_embs, hop_repr)     # [R] — dot-product scores
                # Map action index to beam width k
                k = {0: 5, 1: 10, 2: 50}.get(action, 5)

                # ── Reachability pruning ───────────────────────────────────────
                # Collect all relations that actually exist in the current
                # frontier to avoid following hallucinated (impossible) edges.
                reachable_rels = set()
                with self.env.begin() as txn:
                    for mid in current:
                        f_data = txn.get(f"f:{mid}".encode('utf-8'))
                        if f_data:
                            for r, _ in pickle.loads(f_data):
                                if r in self.rel2id: reachable_rels.add(self.rel2id[r])
                        b_data = txn.get(f"b:{mid}".encode('utf-8'))
                        if b_data:
                            for r, _ in pickle.loads(b_data):
                                if r in self.rel2id: reachable_rels.add(self.rel2id[r])

                # Take top-k semantically similar relations; intersect with reachable.
                # Fall back to the raw top-k if none are reachable (edge case).
                top_k      = torch.topk(sims, k).indices.tolist()
                valid_beam = [rid for rid in top_k if rid in reachable_rels]
                active     = [self.id2rel[rid] for rid in (valid_beam or top_k[:k])]
                execution_log.append(active)

                # Expand frontier by one hop along the selected relations
                next_ents = self.kg_lookup(current, active)
                if not next_ents: break  # dead end — no reachable entities
                current = next_ents

            # ── Annotate final frontier ───────────────────────────────────────
            # Each candidate entity is tagged with its human-readable name and a
            # boolean gold label that F2/F3 training will use as supervision.
            candidates = []
            for mid in current:
                name = self.mid2name.get(mid, "Unknown")
                candidates.append({'mid': mid, 'name': name, 'is_gold': mid in gold})

            results.append({
                'question'  : q,
                'path'      : execution_log,   # hop-level relation beams (context for F2/F3)
                'candidates': candidates
            })

        # ── Persist collected candidates ──────────────────────────────────────
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"[Collector] Saved {len(results)} samples to {output_file}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collector = CDSDataCollector(os.path.join(ROOT, 'checkpoints/exp15_strl_epoch_19.pt'))
    # Collect a subset for training the ranker (faster for now)
    collector.collect(os.path.join(ROOT, 'data/cwq_train.json'), os.path.join(ROOT, 'data/exp16_cds_train.json'), limit=2000)
    collector.collect(os.path.join(ROOT, 'data/cwq_dev.json'), os.path.join(ROOT, 'data/exp16_cds_dev.json'))
