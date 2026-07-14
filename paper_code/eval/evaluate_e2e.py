"""
evaluate_e2e.py — End-to-End KGQA Planning Evaluation Pipeline
===============================================================

Overview
--------
Evaluates all relation-path planning models (Exp 0 through Exp 9) against
the CWQ dev and test splits using a *path-match* protocol:

    question → model predicts relation path → path matched vs gold SPARQL →
    answer correctness inferred without executing the KG

This is equivalent to the standard KGQA planning evaluation used by DRKG,
DAMR, and Plan-Then-Retrieve, which assume correct KG execution given the
correct path.

Paper Results Produced
----------------------
- Table 2  : Per-model Hits@1 / Hits@3 / Hop Accuracy on Dev and Test splits
- Table 3  : Per-hop breakdown (1-hop, 2-hop, 3-hop, 4-hop)

Evaluation Protocol
-------------------
For each CWQ question:
  1. Extract the gold relation path from the question's SPARQL query.
  2. Use the trained model to predict a relation path.
  3. If the predicted (greedy top-1) path matches the gold path → Hit@1 = 1.
  4. Aggregate Hits@1, Hits@3, Per-Hop Accuracy over all questions.

  Note: Hits@1 here measures *planning accuracy* (does the model predict the
  correct relation sequence?), which upper-bounds final answer accuracy.

Inputs
------
- data/cwq_dev.json, data/cwq_test.json, data/cwq_train.json
    Raw CWQ 1.1 dataset splits.
- data/processed_entity/relation2id.pt
    Mapping {relation_name: integer_id}.
- data/processed_entity/train_relations.pt, train_domains.pt
    Pre-processed label tensors used to determine vocabulary sizes.
- checkpoints/exp{N}_*.pt
    Serialised model state-dicts for each experiment.

Outputs
-------
- results.md            Markdown table of all model results + published baselines.
- stdout                Live progress bars and per-hop breakdowns.

Usage
-----
    python eval/evaluate_e2e.py

Evaluation Protocol Note
------------------------
This script does NOT execute SPARQL against Freebase.  Path-match accuracy is
a strong proxy for answer correctness under the assumption of faithful KG
coverage — the same assumption made by all planning-based KGQA papers.
"""
import json, os, sys, re, functools, torch

# ──────────────────────────────────────────────────────
#  Windows encoding fix — ensure UTF-8 output on cp1252 terminals
# ──────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import BertTokenizer

# ──────────────────────────────────────────────────────
#  Project root discovery — allows imports from sibling packages
# ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.sparql_parser import find_reasoning_path
from shared.metrics import hits_at_k

# ============================================================
#  Extract Question-Level Gold Data
# ============================================================

def extract_evaluation_data(cwq_data, relation2id):
    """Extract question-level data for evaluation.

    Parses each CWQ item's SPARQL query to recover the gold relation path,
    filters out questions whose relations are outside the training vocabulary,
    and returns a list of structured sample dicts.

    Args:
        cwq_data    : list of CWQ dataset items (loaded from JSON).
        relation2id : dict mapping relation name → integer id.

    Returns:
        samples  : list of dicts with keys:
                     'id', 'question', 'gold_path', 'gold_answers', 'num_hops'
        (skipped questions are printed to stdout)
    """
    samples = []
    skipped = 0
    
    for item in cwq_data:
        question = item['question']
        sparql = item['sparql']
        
        # Parse the SPARQL query to extract the ordered relation path
        path = find_reasoning_path(sparql)
        if path is None:
            # Skip questions whose SPARQL cannot be parsed into a relation path
            skipped += 1
            continue
        
        # ── Build gold relation path ──────────────────────────────
        # Each hop in `path` is a 4-tuple: (node, relation, direction, next_node)
        gold_relations = []
        all_in_vocab = True
        for node, rel, direction, next_node in path:
            if rel not in relation2id:
                # Relation unseen during training → cannot evaluate this sample
                all_in_vocab = False
                break
            gold_relations.append({
                'relation': rel,
                'relation_id': relation2id[rel],  # integer label used by classifier
                'direction': direction,            # forward / backward traversal
            })
        
        if not all_in_vocab or not gold_relations:
            skipped += 1
            continue
        
        # ── Collect gold answers (used for F1, not path-match) ───
        gold_answers = []
        if 'answers' in item and item['answers']:
            for ans in item['answers']:
                gold_answers.append(ans.get('answer', ''))
        
        samples.append({
            'id': item.get('ID', ''),
            'question': question,
            'gold_path': gold_relations,      # ordered list of hop dicts
            'gold_answers': gold_answers,
            'num_hops': len(gold_relations),  # depth of the gold path (1–4)
        })
    
    print(f"  Extracted {len(samples)} samples, skipped {skipped}")
    return samples

# ──────────────────────────────────────────────────────
#  Model Prediction Functions
#  (one per experiment architecture)
# ──────────────────────────────────────────────────────

def predict_exp0(model, tokenizer, question, device, num_rels, k=10):
    """Exp 0: flat relation classifier — returns top-k relation IDs.

    The flat baseline treats all hops as a single multi-class classification
    over the full relation vocabulary.  The returned list is used in order:
    position h → predicted relation for hop h.
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    # Softmax → top-k indices (descending probability)
    probs = F.softmax(logits, dim=-1)
    _, topk = torch.topk(probs, k=k, dim=-1)
    return topk[0].cpu().tolist()

def predict_exp3(model, tokenizer, question, device, k=10):
    """Exp 3 (PCT): Progressive Constraint Tightening — top-k + confidence.

    PCT predicts domain first, then relations.  We return the top-k relation
    IDs and the model's overall confidence scalar (used for analysis only).
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Model output: (domain_logits, domain_ids, rel_logits, confidence)
            _, _, rel_logits, confidence = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    probs = F.softmax(rel_logits, dim=-1)
    _, topk = torch.topk(probs, k=k, dim=-1)
    return topk[0].cpu().tolist(), confidence[0].item()

def predict_exp4(model, tokenizer, question, device, max_hops=4, k=10):
    """Exp 4 (CHCP): Cross-Hop Coherence Planning — per-hop top-k + stop prob.

    CHCP produces independent per-hop relation logits and a stop signal.
    We decode up to `max_hops` steps and return, for each hop, the top-k
    predicted relation IDs and the stop probability.
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # rel_logits : [B, max_hops, num_relations]
            # stop_logits: [B, max_hops]
            rel_logits, stop_logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        # sigmoid converts scalar stop logit → probability of stopping at this hop
        stop_p = torch.sigmoid(stop_logits[0, h]).item()
        results.append({'top_ids': topk.cpu().tolist(), 'stop_prob': stop_p})
    return results

def predict_exp6(model, tokenizer, question, device, max_hops=4, k=10):
    """Exp 6 (Unified): Unified Adaptive-CHCP — per-hop top-k + stop prob.

    Same decoding logic as Exp 4 but the model uses domain-aware attention and
    returns outputs as a dict with keys 'rel_logits' and 'stop_logits'.
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            rel_logits = out['rel_logits']    # [B, max_hops, num_relations]
            stop_logits = out['stop_logits']  # [B, max_hops]
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        stop_p = torch.sigmoid(stop_logits[0, h]).item()
        results.append({'top_ids': topk.cpu().tolist(), 'stop_prob': stop_p})
    return results

def predict_exp7(model, tokenizer, question, device, max_hops=4, k=10):
    """Exp 7 (Scaled RoBERTa): per-hop top-k predictions from RoBERTa-Large backbone.

    Architecturally identical to Exp 6 but uses a RoBERTa-Large encoder instead
    of BERT, boosting the relation representation quality.
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            rel_logits = out['rel_logits']
            stop_logits = out['stop_logits']
    results = []
    for h in range(max_hops):
        probs = F.softmax(rel_logits[0, h], dim=-1)
        _, topk = torch.topk(probs, k=k, dim=-1)
        stop_p = torch.sigmoid(stop_logits[0, h]).item()
        results.append({'top_ids': topk.cpu().tolist(), 'stop_prob': stop_p})
    return results

def predict_exp9(rl_agent, tokenizer, question, device, max_hops=4, k=10):
    """Exp 9: RL Meta-Constraint Agent (RLMC).

    The RL agent predicts a *constraint action* per hop that controls beam width:
      TIGHT  (action=0) → beam width 1   (very precise)
      MEDIUM (action=1) → beam width 5
      LOOSE  (action=2) → beam width 50  (broad recall)
      STOP   (action=3) → terminate path

    The returned list mirrors the CHCP format so `evaluate_model` can use it
    unchanged with model_type='rlmc'.
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # action_logits: [B, max_hops, 4]  — RL policy head
            # rel_logits   : [B, max_hops, num_relations] — relation scorer
            action_logits, _, rel_logits, _ = rl_agent(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            
    results = []
    # Greedy action decoding: argmax over the 4 action classes per hop
    actions = torch.argmax(action_logits[0], dim=-1).tolist()
    
    for h in range(max_hops):
        a = actions[h]
        probs = F.softmax(rel_logits[0, h], dim=-1)
        
        # ── Map RL action → candidate beam width ──────────────
        if a == 0:   # TIGHT  → single top relation
            w = 1
        elif a == 1: # MEDIUM → top-5 relations
            w = 5
        elif a == 2: # LOOSE  → top-50 relations (domain-wide recall)
            w = 50
        else:        # STOP   → no more hops
            w = 0
            
        if w > 0:
            _, topw = torch.topk(probs, k=w, dim=-1)
            # Store the constraint-width candidates for path-match evaluation
            results.append({'top_ids': topw.cpu().tolist()})
        else:
            break  # STOP action: truncate path here
            
    return results

def predict_exp10_reranked(rl_agent, reranker, tokenizer, question, device, id2rel, max_hops=4):
    """Exp 10: Candidate path reranking with a Cross-Encoder.

    Two-stage prediction:
      1. Generate a small candidate set of relation paths using the RL agent.
      2. Score each candidate path with a cross-encoder (question, path → score).
      3. Return the highest-scoring path as width=1 top_ids per hop.

    The cross-encoder takes as input:
        [CLS] question [SEP] "rel1 -> rel2 -> ..." [SEP]
    and outputs a relevance logit.
    """
    # ── Stage 1: Generate candidate paths via RL agent ────────
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        action_logits, _, rel_logits, _ = rl_agent(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    
    actions = torch.argmax(action_logits[0], dim=-1).tolist()
    L = 4  # Evaluate full 4-hop paths regardless of STOP signal
    
    # Approximate beam search: take top-5 relation at each hop, form 5 paths
    # by pairing rank-r relation at hop 0 with rank-r relation at hop 1, etc.
    logits = rel_logits[0, :L]
    top5_per_hop = torch.topk(logits, 5, dim=-1).indices.tolist()
    
    candidates = set()
    for rank in range(5):
        # Diagonal candidate: same rank across all hops
        candidates.add(tuple([top5_per_hop[h][rank] for h in range(L)]))
        
    # ── Stage 2: Score each candidate path with cross-encoder ─
    best_path = None
    best_score = -float('inf')
    
    for path in candidates:
        # Convert integer IDs back to readable relation strings for the encoder
        path_str = " -> ".join([id2rel[r] for r in path])
        # Cross-encoder: [CLS] question [SEP] path [SEP]
        enc_c = tokenizer(question, path_str, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            # Cross-encoder returns a single relevance logit at index [0, 0]
            score = reranker(enc_c['input_ids'].to(device), enc_c['attention_mask'].to(device)).logits[0, 0].item()
            
        if score > best_score:
            best_score = score
            best_path = list(path)
            
    # ── Format output to match per-hop structure ───────────────
    # Return as width=1 top_ids so freebase_execution_eval can run it easily
    return [{'top_ids': [best_path[h]]} for h in range(len(best_path))]

def predict_exp9_sota(rl_agent, tokenizer, question, device, kg, topic_mid, id2rel, beam_width=5, max_hops=4):
    """SOTA Inference: Graph-Constrained Beam Search (experimental / incomplete).

    Only allows relation transitions that physically exist in the KG from the
    current set of frontier entities — this constrains the search space to
    reachable paths and dramatically improves precision.

    NOTE: This function is a work-in-progress stub.  Full implementation
    requires rel2id to be passed in; the loop body is not yet complete.
    """
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        _, _, rel_logits, _ = rl_agent(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    
    # Beam state: (current_frontier_entities, path_so_far, cumulative_log_prob)
    beam = [({topic_mid}, [], 0.0)]
    
    for h in range(max_hops):
        new_beam = []
        # Per-hop log-probabilities from the relation scorer
        hop_probs = F.softmax(rel_logits[0, h], dim=-1)
        hop_log_probs = torch.log(hop_probs + 1e-10)
        
        for current_ents, path, score in beam:
            # ── Collect all relations reachable from frontier entities ──
            reachable_rels = set()
            for ent in current_ents:
                for r, _ in kg.forward.get(ent, []):
                    reachable_rels.add(r)   # forward edge: ent -[r]-> target
                for r, _ in kg.backward.get(ent, []):
                    reachable_rels.add(r)   # backward edge: source -[r]-> ent
            
            if not reachable_rels:
                continue  # dead end — this beam branch cannot continue
                
            # ── Score only reachable relations ─────────────────────────
            for rel_name in reachable_rels:
                if rel_name not in rl_agent.base_planner.tokenizer.get_vocab():
                     continue  # safety: relation not in encoder vocab
                # NOTE: rel2id lookup is needed here — currently a placeholder
                pass
        # rel2id is needed inside the beam loop — to be passed as a parameter
    
    # Stub return: full implementation deferred
    return None


def evaluate_model(samples, model, tokenizer, id2relation, device,
                   model_name, predict_fn, model_type='flat'):
    """Evaluate a single model on a set of CWQ samples using path-match.

    For each question:
      - The model predicts relation(s) for each hop.
      - The predicted path is compared against the gold path.
      - A full match at position 0 (top-1) counts toward Hits@1.
      - A match within the top-3 predictions counts toward Hits@3.

    Supports two prediction shapes:
      flat  — predict_fn returns a single ranked list of relation IDs
              (used for Exp 0 and Exp 3 which share one softmax over all hops)
      chcp / unified / roberta / rlmc
            — predict_fn returns a list of per-hop dicts with 'top_ids'

    Metrics computed:
      Hits@1       : fraction of questions with full top-1 path match
      Hits@3       : fraction of questions with gold path in top-3 combinations
      Hop Accuracy : fraction of individual hops predicted correctly (top-1)

    Args:
        samples     : list of dicts from extract_evaluation_data.
        model       : loaded PyTorch model (eval mode).
        tokenizer   : HuggingFace tokenizer compatible with the model.
        id2relation : dict mapping integer id → relation name string.
        device      : torch.device.
        model_name  : display string for progress bars and output.
        predict_fn  : callable matching the model's prediction API.
        model_type  : one of 'flat', 'chcp', 'unified', 'roberta', 'rlmc'.

    Returns:
        results dict with keys:
          'model', 'total', 'hits@1', 'hits@3', 'hop_accuracy', 'by_hops'
    """
    total = 0
    hits1 = 0        # number of questions with full top-1 path match
    hits3 = 0        # number of questions with gold path in top-3 per hop
    hop_correct = 0  # number of individual hops predicted correctly
    hop_total = 0    # total number of hops evaluated across all questions
    
    # Aggregate per-depth statistics for the hop breakdown table
    by_hops = defaultdict(lambda: {'total': 0, 'hits1': 0, 'hop_correct': 0, 'hop_total': 0})
    
    for sample in tqdm(samples, desc=f"Eval {model_name}"):
        question = sample['question']
        gold_path = sample['gold_path']
        num_hops = sample['num_hops']
        
        if model_type == 'flat':
            # ── Flat model: single ranked list shared across all hops ──
            if model_type == 'flat' and model_name.startswith('Exp 0'):
                # Exp 0: flat classifier — pass vocab size explicitly
                top_ids = predict_fn(model, tokenizer, question, device, len(id2relation), k=10)
            else:
                # Exp 3: PCT — returns (top_ids, confidence); ignore confidence here
                top_ids, conf = predict_fn(model, tokenizer, question, device, k=10)
            
            # Assign consecutive rank positions to consecutive hops
            # e.g., hop 0 → top_ids[0], hop 1 → top_ids[1], ...
            path_match_1 = True   # tracks whether top-1 is correct for ALL hops
            path_match_3 = True   # tracks whether gold is in top-3 for ALL hops
            for h, gold_hop in enumerate(gold_path):
                gold_id = gold_hop['relation_id']
                hop_total += 1
                by_hops[num_hops]['hop_total'] += 1
                
                if h < len(top_ids):
                    pred_id = top_ids[h]
                    if pred_id == gold_id:
                        hop_correct += 1
                        by_hops[num_hops]['hop_correct'] += 1
                    else:
                        path_match_1 = False  # top-1 path broken at hop h
                    # Check if gold appears in a 3-wide window starting at hop h
                    if gold_id not in top_ids[max(0, h):h+3]:
                        path_match_3 = False
                else:
                    # Ran out of predictions before exhausting all hops
                    path_match_1 = False
                    path_match_3 = False
                    
        elif model_type in ['chcp', 'unified', 'roberta', 'rlmc']:
            # ── Structured model: per-hop top-k dicts ─────────────────
            hop_preds = predict_fn(model, tokenizer, question, device, max_hops=4, k=10)
            
            path_match_1 = True
            path_match_3 = True
            for h, gold_hop in enumerate(gold_path):
                gold_id = gold_hop['relation_id']
                hop_total += 1
                by_hops[num_hops]['hop_total'] += 1
                
                if h < len(hop_preds):
                    pred_top = hop_preds[h]['top_ids']  # ranked list for this hop
                    # Top-1 check: first prediction matches gold?
                    if pred_top[0] == gold_id:
                        hop_correct += 1
                        by_hops[num_hops]['hop_correct'] += 1
                    else:
                        path_match_1 = False
                    # Top-3 check: gold appears anywhere in first 3 predictions?
                    if gold_id not in pred_top[:3]:
                        path_match_3 = False
                else:
                    # Model terminated the path early (fewer hops than gold)
                    path_match_1 = False
                    path_match_3 = False
        
        # ── Accumulate path-level outcomes ────────────────────────
        if path_match_1:
            hits1 += 1
            by_hops[num_hops]['hits1'] += 1
        if path_match_3:
            hits3 += 1
        
        total += 1
        by_hops[num_hops]['total'] += 1
    
    # ── Compute final metric fractions ────────────────────────────
    results = {
        'model': model_name,
        'total': total,
        'hits@1': hits1 / total if total > 0 else 0,
        'hits@3': hits3 / total if total > 0 else 0,
        'hop_accuracy': hop_correct / hop_total if hop_total > 0 else 0,
        'by_hops': dict(by_hops),
    }
    
    # ── Print per-depth breakdown ──────────────────────────────────
    print(f"\n  {model_name} Results:")
    print(f"    Overall Hits@1: {results['hits@1']:.4f} | Hits@3: {results['hits@3']:.4f} | Hop Acc: {results['hop_accuracy']:.4f}")
    for nh in sorted(by_hops.keys()):
        bh = by_hops[nh]
        h1 = bh['hits1']/bh['total'] if bh['total'] > 0 else 0
        ha = bh['hop_correct']/bh['hop_total'] if bh['hop_total'] > 0 else 0
        print(f"    {nh}-hop: Hits@1={h1:.4f} | Hop Acc={ha:.4f} | ({bh['total']} questions)")
    
    return results

# ──────────────────────────────────────────────────────
#  Main — Load Data, Evaluate All Experiments, Write Results
# ──────────────────────────────────────────────────────

def main():
    # ── Device setup ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ── Load relation vocabulary ───────────────────────────────────
    data_dir = os.path.join(ROOT, 'data/processed_entity')
    relation2id = torch.load(os.path.join(data_dir, 'relation2id.pt'))
    id2relation = {v: k for k, v in relation2id.items()}  # inverse lookup
    
    # ── Extract evaluation samples from all splits ─────────────────
    print("\n[1/3] Extracting evaluation data...")
    dev_data   = json.load(open(os.path.join(ROOT, 'data/cwq_dev.json'),   'r', encoding='utf-8'))
    test_data  = json.load(open(os.path.join(ROOT, 'data/cwq_test.json'),  'r', encoding='utf-8'))
    train_data = json.load(open(os.path.join(ROOT, 'data/cwq_train.json'), 'r', encoding='utf-8'))
    
    print("  Dev set:")
    dev_samples = extract_evaluation_data(dev_data, relation2id)
    print("  Test set:")
    test_samples = extract_evaluation_data(test_data, relation2id)
    print("  Train set:")
    train_samples = extract_evaluation_data(train_data, relation2id)
    
    datasets = [
        ("Dev", dev_samples),
        ("Test", test_samples)
    ]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_results = []
    
    # ──────────────────────────────────────────────────────
    #  Experiment Evaluation Loop
    #  Each block: load checkpoint → evaluate → free GPU memory
    # ──────────────────────────────────────────────────────
    print("\n[2/3] Loading and evaluating models...")

    # ── Exp 0: Flat BERT Baseline ─────────────────────────────────
    # Single softmax over all relations; no hop-aware structure.
    print("\n  ---- Exp 0: Flat BERT Baseline ----")
    from train.exp0_flat_baseline import BERTRelationClassifier
    train_r = torch.load(os.path.join(data_dir, 'train_relations.pt'))
    num_rel = int(torch.max(train_r).item()) + 1  # vocab size from training labels
    model = BERTRelationClassifier(num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp0_relation_flat_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r0 = evaluate_model(s_data, model, tokenizer, id2relation, device, f"Exp 0 ({s_name})", predict_exp0, model_type='flat')
        all_results.append(r0)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 3: Progressive Constraint Tightening (PCT) ────────────
    # Domain classifier + relation classifier; confidence-gated.
    print("\n  ---- Exp 3: Progressive Constraint Tightening ----")
    from train.exp3_pct import PCTModel
    train_d = torch.load(os.path.join(data_dir, 'train_domains.pt'))
    num_dom = int(torch.max(train_d).item()) + 1  # number of domain classes
    model = PCTModel(num_domains=num_dom, num_relations=num_rel).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp3_pct_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r3 = evaluate_model(s_data, model, tokenizer, id2relation, device, f"Exp 3 ({s_name})", predict_exp3, model_type='flat')
        all_results.append(r3)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 4: Cross-Hop Coherence Planning (CHCP) ────────────────
    # Per-hop relation heads share BERT encoder; stop signal per hop.
    print("\n  ---- Exp 4: Cross-Hop Coherence Planning ----")
    from train.exp4_chcp import CHCPModel
    rel2id_full = torch.load(os.path.join(data_dir, 'relation2id.pt'))
    num_rel_full = len(rel2id_full)   # full vocabulary (may differ from flat baseline)
    id2rel_full = {v: k for k, v in rel2id_full.items()}
    model = CHCPModel(num_relations=num_rel_full, max_hops=4).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp4_chcp_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r4 = evaluate_model(s_data, model, tokenizer, id2rel_full, device, f"Exp 4 ({s_name})", predict_exp4, model_type='chcp')
        all_results.append(r4)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 4-RL: Reinforcement-learned CHCP ──────────────────────
    # Same architecture as Exp 4 but fine-tuned with PPO reward signal.
    print("\n  ---- Exp 4-RL: Reinforcement Learned CHCP ----")
    model = CHCPModel(num_relations=num_rel_full, max_hops=4).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp4_rl_epoch_49.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r4rl = evaluate_model(s_data, model, tokenizer, id2rel_full, device, f"Exp 4-RL ({s_name})", predict_exp4, model_type='chcp')
        all_results.append(r4rl)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 6: Unified Adaptive-CHCP ──────────────────────────────
    # Domain-conditioned CHCP with adaptive hop gating.
    print("\n  ---- Exp 6: Unified Adaptive-CHCP ----")
    from train.exp6_unified import UnifiedKGQAPlanner
    train_d = torch.load(os.path.join(data_dir, 'train_domains.pt'))
    num_dom = int(torch.max(train_d).item()) + 1
    model = UnifiedKGQAPlanner(num_dom, num_rel_full).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp6_unified_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r6 = evaluate_model(s_data, model, tokenizer, id2rel_full, device, f"Exp 6 ({s_name})", predict_exp6, model_type='unified')
        all_results.append(r6)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 7: Scaled RoBERTa-Large ───────────────────────────────
    # Switches backbone to RoBERTa-Large for stronger contextual representations.
    print("\n  ---- Exp 7: Scaled RoBERTa-Large ----")
    from train.exp7_roberta import ScaledUnifiedPlanner
    from transformers import RobertaTokenizer
    rob_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = ScaledUnifiedPlanner(num_dom, num_rel_full).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_epoch_29.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r7 = evaluate_model(s_data, model, rob_tokenizer, id2rel_full, device, f"Exp 7 ({s_name})", predict_exp7, model_type='roberta')
        all_results.append(r7)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 8: Contrastive RoBERTa (CPD) ──────────────────────────
    # Same architecture as Exp 7 but trained with a contrastive path-distance
    # objective that pushes incorrect paths away from the question embedding.
    print("\n  ---- Exp 8: Contrastive RoBERTa (CPD) ----")
    model = ScaledUnifiedPlanner(num_dom, num_rel_full).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp8_cpd_best.pt'), map_location=device))
    model.eval()
    for s_name, s_data in datasets:
        r8 = evaluate_model(s_data, model, rob_tokenizer, id2rel_full, device, f"Exp 8 ({s_name})", predict_exp7, model_type='roberta')
        all_results.append(r8)
    del model; torch.cuda.empty_cache()
    
    # ── Exp 9: RL Meta-Constraint Agent (RLMC) ────────────────────
    # RoBERTa-Large base + RL policy that dynamically selects beam width per hop.
    # The base planner weights are loaded first, then the RL wrapper is stacked.
    print("\n  ---- Exp 9: RL Meta-Constraint Agent (RLMC) ----")
    from train.exp9_rlmc import RLConstraintAgent
    base_model = ScaledUnifiedPlanner(num_dom, num_rel_full).to(device)
    base_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    
    rl_model = RLConstraintAgent(base_model).to(device)
    rl_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=device))
    rl_model.eval()
    for s_name, s_data in datasets:
        r9 = evaluate_model(s_data, rl_model, rob_tokenizer, id2rel_full, device, f"Exp 9 ({s_name})", predict_exp9, model_type='rlmc')
        all_results.append(r9)
    del rl_model; torch.cuda.empty_cache()
    
    # ──────────────────────────────────────────────────────
    #  Write Results to Markdown
    # ──────────────────────────────────────────────────────
    print("\n[3/3] Writing results...")
    rp = os.path.join(ROOT, 'results.md')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write("# KGQA Research Experiment Results\n\n")
        f.write(f"## End-to-End Evaluation\n\n")
        f.write("Evaluation protocol: question → model predicts relation path → path match against gold SPARQL → derive answer correctness.\n")
        f.write("This matches the planning evaluation used by DRKG, DAMR, and Plan-Then-Retrieve.\n\n")
        
        # ── Main results table ─────────────────────────────────────
        f.write("| Model | Hits@1 | Hits@3 | Hop Accuracy | Questions |\n")
        f.write("|---|---|---|---|---|\n")
        for r in all_results:
            f.write(f"| **{r['model']}** | {r['hits@1']:.4f} | {r['hits@3']:.4f} | {r['hop_accuracy']:.4f} | {r['total']} |\n")
        
        # ── Per-depth breakdown table ──────────────────────────────
        f.write("\n### Breakdown by Number of Hops\n\n")
        f.write("| Model | 1-hop | 2-hop | 3-hop | 4-hop |\n")
        f.write("|---|---|---|---|---|\n")
        for r in all_results:
            row = f"| **{r['model']}** |"
            for nh in range(1, 5):
                bh = r['by_hops'].get(nh, {'hits1': 0, 'total': 0})
                h1 = bh['hits1']/bh['total'] if bh['total'] > 0 else 0
                row += f" {h1:.4f} ({bh.get('total',0)}) |"
            f.write(row + "\n")
        
        # ── Published baselines for comparison ────────────────────
        f.write("\n---\n\n## Comparable Published Results on CWQ\n\n")
        f.write("| Method | Hits@1 | F1 | Year |\n")
        f.write("|---|---|---|---|\n")
        f.write("| NSM | 0.486 | 0.483 | 2021 |\n")
        f.write("| SR+NSM | 0.505 | - | 2022 |\n")
        f.write("| TIARA | 0.534 | - | 2022 |\n")
        f.write("| ChatKBQA | 0.555 | - | 2024 |\n")
        f.write("| DRKG | 0.669 | - | 2025 |\n")
        
        f.write("\n> **Note**: Our evaluation uses path-matching on a CWQ-derived subgraph.\n")
        f.write("> Published results use Freebase execution. Direct comparison should be interpreted carefully.\n")
        f.write("> Our Hits@1 measures *planning accuracy* (does the model predict the correct relation path?)\n")
        f.write("> which upper-bounds the final answer accuracy.\n")
        
        f.write("\n---\n\n## Performance Notes\n\n")
        f.write("- **GPU**: RTX 5070 Laptop (SM 12.0 / Blackwell)\n")
        f.write("- **PyTorch**: 2.11.0+cu128 with Mixed Precision (AMP)\n")
        f.write("- **Dataset**: ComplexWebQuestions (CWQ) 1.1\n")
        f.write("- **Evaluation**: Path-match based (planning accuracy)\n")
    
    print(f"\nResults written to {rp}")
    
    # ── Final console summary ──────────────────────────────────────
    print("\n" + "="*60)
    print("  END-TO-END RESULTS SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['model']:30s} | H@1: {r['hits@1']:.4f} | H@3: {r['hits@3']:.4f} | Hop: {r['hop_accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
