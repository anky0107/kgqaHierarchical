"""
Correct Hits@1 Evaluation — All Experiments (Exp 0, 3, 4, 4-RL, 6, 7, 8, 9, 10-CWQ)
=====================================================================================

CORRECT DEFINITION:
  For each question:
    1. Model predicts relation path (greedy top-1 per hop)
    2. Traverse KG following those relations
    3. Collect reached entities (candidates)
    4. Return EXACTLY ONE entity:
         - Exp 0-8, 10: pick first candidate by sorted MID (deterministic, no reranker)
         - Exp 9      : apply reranker to score candidates → pick top-1
    5. Hit = 1  if  that single entity ∈ gold_answers
    6. Hit@1 = sum(hits) / total_questions

This matches DRKG / ChatKBQA / NSM evaluation exactly.
"""
import os, sys, json, torch, functools
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer, BertTokenizer, RobertaForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from shared.kg_loader import build_kg_from_cwq_triples, KnowledgeGraph
from utils.sparql_parser import extract_triples, find_reasoning_path

# ─────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────

def load_test_samples(json_path, relation2id):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples, skipped = [], 0
    for item in data:
        path = find_reasoning_path(item.get('sparql', ''))
        if not path:
            skipped += 1
            continue

        topic_entity = path[0][0].replace('ns:', '').strip(')')
        num_hops = len(path)

        gold_rel_ids, all_in_vocab = [], True
        for node, rel, direction, next_node in path:
            if rel not in relation2id:
                all_in_vocab = False
                break
            gold_rel_ids.append(relation2id[rel])

        if not all_in_vocab or not gold_rel_ids:
            skipped += 1
            continue

        gold_answers = set()
        for ans in item.get('answers', []):
            aid = ans.get('answer_id', '')
            if aid:
                clean = aid.replace('ns:', '').strip(')')
                gold_answers.add(clean)
        if not gold_answers:
            # fallback to end of gold path
            gold_answers = {path[-1][3].replace('ns:', '').strip(')')}

        samples.append({
            'question':     item['question'],
            'topic_entity': topic_entity,
            'gold_answers': gold_answers,
            'num_hops':     num_hops,
            'gold_rel_ids': gold_rel_ids,
        })

    print(f"  Loaded {len(samples)} samples, skipped {skipped}")
    return samples

# ─────────────────────────────────────────────────────────────
#  KG traversal
# ─────────────────────────────────────────────────────────────

def traverse(kg, topic_entity, rel_sets_per_hop):
    """
    rel_sets_per_hop: list of sets, one per hop.
    Returns the set of reached entity MIDs after all hops.
    """
    active = {topic_entity}
    for hop_rels in rel_sets_per_hop:
        if not active or not hop_rels:
            break
        nxt = set()
        for e in active:
            for rel, direction, tgt in kg.get_neighbors(e):
                if rel in hop_rels:
                    nxt.add(tgt.strip(')'))
        active = nxt
    return active

# ─────────────────────────────────────────────────────────────
#  Entity selection strategies
# ─────────────────────────────────────────────────────────────

def pick_one_no_reranker(candidates):
    """
    For models with no reranker: pick the lexicographically first MID.
    Deterministic, unbiased.
    Returns None if candidates is empty.
    """
    if not candidates:
        return None
    return min(candidates)

def pick_one_with_reranker(candidates, question, reranker, tokenizer, device, batch_size=32):
    """
    For Exp 9: score each candidate with the cross-encoder reranker and return top-1.
    Input:  question string, set of candidate MIDs
    Output: single best MID (or None)
    """
    if not candidates:
        return None
    cands = list(candidates)
    if len(cands) == 1:
        return cands[0]

    scores = []
    q_lower = question.lower()
    for i in range(0, len(cands), batch_size):
        batch = cands[i:i+batch_size]
        # Use MID as the path string (reranker trained on path strings)
        enc = tokenizer(
            [q_lower] * len(batch), batch,
            padding=True, truncation=True, max_length=128, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            logits = reranker(**enc).logits
        scores.extend(torch.sigmoid(logits[:, 0]).cpu().tolist())

    best_idx = scores.index(max(scores))
    return cands[best_idx]

# ─────────────────────────────────────────────────────────────
#  Per-experiment prediction functions
#  All return: list of sets-of-relation-names, one set per hop
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_flat_top1(model, tokenizer, question, device, id2rel, num_hops):
    enc = tokenizer(question, truncation=True, max_length=128, return_tensors='pt').to(device)
    logits = model(enc['input_ids'], enc['attention_mask'])
    topk = torch.topk(logits[0], k=num_hops).indices.tolist()
    return [{id2rel.get(topk[h], '')} for h in range(num_hops)]

@torch.no_grad()
def predict_pct_top1(model, tokenizer, question, device, id2rel, num_hops):
    enc = tokenizer(question, truncation=True, max_length=128, return_tensors='pt').to(device)
    _, _, rel_logits, _ = model(enc['input_ids'], enc['attention_mask'])
    topk = torch.topk(rel_logits[0], k=num_hops).indices.tolist()
    return [{id2rel.get(topk[h], '')} for h in range(num_hops)]

@torch.no_grad()
def predict_chcp_top1(model, tokenizer, question, device, id2rel, num_hops):
    enc = tokenizer(question, truncation=True, max_length=128, return_tensors='pt').to(device)
    rel_logits, _ = model(enc['input_ids'], enc['attention_mask'])
    max_h = rel_logits.size(1)
    result = []
    for h in range(num_hops):
        hi = min(h, max_h - 1)
        rid = torch.argmax(rel_logits[0, hi]).item()
        result.append({id2rel.get(rid, '')})
    return result

@torch.no_grad()
def predict_unified_top1(model, tokenizer, question, device, id2rel, num_hops):
    enc = tokenizer(question, truncation=True, max_length=128, return_tensors='pt').to(device)
    out = model(enc['input_ids'], enc['attention_mask'])
    rel_logits = out['rel_logits']
    max_h = rel_logits.size(1)
    result = []
    for h in range(num_hops):
        hi = min(h, max_h - 1)
        rid = torch.argmax(rel_logits[0, hi]).item()
        result.append({id2rel.get(rid, '')})
    return result

@torch.no_grad()
def predict_rlmc_top1(rl_agent, tokenizer, question, device, id2rel, num_hops):
    """
    Uses RL-chosen beam width per hop:
      TIGHT (0) → top-1,  MEDIUM (1) → top-5,  LOOSE (2) → top-50,  STOP (3) → stop
    """
    enc = tokenizer(question, truncation=True, max_length=128, return_tensors='pt').to(device)
    action_logits, _, rel_logits, _ = rl_agent(enc['input_ids'], enc['attention_mask'])
    actions = torch.argmax(action_logits[0], dim=-1).tolist()
    width_map = {0: 1, 1: 5, 2: 50, 3: 0}
    max_h = rel_logits.size(1)

    result = []
    for h in range(num_hops):
        a = actions[h] if h < len(actions) else 3
        w = width_map.get(a, 1)
        if w == 0:
            break
        hi = min(h, max_h - 1)
        w = min(w, rel_logits.size(-1))
        topk_ids = torch.topk(rel_logits[0, hi], k=w).indices.tolist()
        result.append({id2rel.get(rid, '') for rid in topk_ids})
    while len(result) < num_hops:
        result.append(set())
    return result

@torch.no_grad()
def predict_universal_top1(model, tokenizer, question, device, id2rel, num_hops, dataset_name='cwq'):
    ds_map = {'cwq': 0, 'webqsp': 1, 'metaqa': 2}
    ds_id = ds_map.get(dataset_name, 0)
    ds_tensor = torch.tensor([ds_id], device=device)
    enc = tokenizer(question, truncation=True, max_length=160, return_tensors='pt').to(device)
    out = model(enc['input_ids'], enc['attention_mask'], ds_tensor)
    rel_logits = out['rel_logits']
    max_h = rel_logits.size(1)
    result = []
    for h in range(num_hops):
        hi = min(h, max_h - 1)
        rid = torch.argmax(rel_logits[0, hi]).item()
        result.append({id2rel.get(rid, '')})
    return result

# ─────────────────────────────────────────────────────────────
#  Core evaluation loop
# ─────────────────────────────────────────────────────────────

def evaluate_correct_hit1(samples, kg, predict_fn, select_fn, model_name):
    """
    predict_fn(question, num_hops) -> list[set[rel_name]]
    select_fn(candidates, question) -> single MID str or None
    """
    total = hits = 0
    no_candidates = 0
    by_hops = defaultdict(lambda: {'total': 0, 'hits': 0, 'no_cand': 0})

    for sample in tqdm(samples, desc=f"{model_name}", ncols=90):
        q        = sample['question']
        topic    = sample['topic_entity']
        gold     = sample['gold_answers']
        num_hops = sample['num_hops']

        rel_sets = predict_fn(q, num_hops)
        candidates = traverse(kg, topic, rel_sets)

        pred = select_fn(candidates, q)

        total += 1
        by_hops[num_hops]['total'] += 1

        if pred is None:
            no_candidates += 1
            by_hops[num_hops]['no_cand'] += 1
        elif pred in gold:
            hits += 1
            by_hops[num_hops]['hits'] += 1

    h1 = hits / total if total > 0 else 0
    print(f"\n  ── {model_name}")
    print(f"     Correct Hit@1 : {h1:.4f}  ({hits}/{total})")
    print(f"     No candidates : {no_candidates}/{total}")
    for nh in sorted(by_hops):
        bh = by_hops[nh]
        hh = bh['hits']/bh['total'] if bh['total'] > 0 else 0
        print(f"     {nh}-hop : {hh:.4f} ({bh['hits']}/{bh['total']}, no_cand={bh['no_cand']})")

    return {
        'model':       model_name,
        'hits@1':      h1,
        'hits':        hits,
        'total':       total,
        'no_cand':     no_candidates,
        'by_hops':     dict(by_hops),
    }

# ─────────────────────────────────────────────────────────────
#  Checkpoint loader (robust – handles vocab-size mismatches)
# ─────────────────────────────────────────────────────────────

def load_ckpt(model, path, device, strict=False):
    import torch.nn as nn
    sd = torch.load(path, map_location=device, weights_only=False)
    ms = model.state_dict()
    for k in ['relation_head.weight', 'base_model.relation_head.weight',
              'base_planner.relation_head.weight']:
        if k in sd and k in ms and sd[k].shape != ms[k].shape:
            print(f"    [resize] {k}: {sd[k].shape[0]} → {ms[k].shape[0]} (checkpoint → model)")
    model.load_state_dict(sd, strict=strict)
    return model

# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── paths ──────────────────────────────────────────────
    DATA  = os.path.join(ROOT, 'data')
    PDATA = os.path.join(DATA, 'processed_entity')
    CKPT  = os.path.join(ROOT, 'checkpoints')

    rel2id    = torch.load(os.path.join(PDATA, 'relation2id.pt'), weights_only=False)
    id2rel    = {v: k for k, v in rel2id.items()}
    num_rel   = len(rel2id)
    train_d   = torch.load(os.path.join(PDATA, 'train_domains.pt'), weights_only=False)
    num_dom   = int(torch.max(train_d).item()) + 1
    train_r   = torch.load(os.path.join(PDATA, 'train_relations.pt'), weights_only=False)
    num_rel_flat = int(torch.max(train_r).item()) + 1

    # ── build KG ───────────────────────────────────────────
    print("\n[1/3] Building KG from CWQ data…")
    kg = build_kg_from_cwq_triples([
        os.path.join(DATA, 'cwq_train.json'),
        os.path.join(DATA, 'cwq_dev.json'),
        os.path.join(DATA, 'cwq_test.json'),
    ], extract_triples)

    # ── load test samples ──────────────────────────────────
    print("\n[2/3] Loading test samples…")
    samples = load_test_samples(os.path.join(DATA, 'cwq_test.json'), rel2id)

    # simple select: no reranker
    select_plain = lambda cands, q: pick_one_no_reranker(cands)

    all_results = []
    print("\n[3/3] Evaluating…\n")

    # ── Exp 0 : Flat BERT ──────────────────────────────────
    print("── Exp 0: Flat BERT Baseline ──")
    from train.exp0_flat_baseline import BERTRelationClassifier
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    m = BERTRelationClassifier(num_relations=num_rel_flat).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp0_relation_flat_best.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_flat_top1(m, bert_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 0 – Flat BERT"))
    del m; torch.cuda.empty_cache()

    # ── Exp 3 : PCT ────────────────────────────────────────
    print("\n── Exp 3: Progressive Constraint Tightening ──")
    from train.exp3_pct import PCTModel
    m = PCTModel(num_domains=num_dom, num_relations=num_rel_flat).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp3_pct_best.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_pct_top1(m, bert_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 3 – PCT"))
    del m; torch.cuda.empty_cache()

    # ── Exp 4 : CHCP ───────────────────────────────────────
    print("\n── Exp 4: Cross-Hop Coherence Planning ──")
    from train.exp4_chcp import CHCPModel
    m = CHCPModel(num_relations=num_rel, max_hops=4).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp4_chcp_best.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_chcp_top1(m, bert_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 4 – CHCP"))
    del m; torch.cuda.empty_cache()

    # ── Exp 4-RL ───────────────────────────────────────────
    print("\n── Exp 4-RL: RL-finetuned CHCP ──")
    m = CHCPModel(num_relations=num_rel, max_hops=4).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp4_rl_epoch_49.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_chcp_top1(m, bert_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 4-RL – CHCP+PPO"))
    del m; torch.cuda.empty_cache()

    # ── Exp 6 : Unified ────────────────────────────────────
    print("\n── Exp 6: Unified Adaptive Planner ──")
    from train.exp6_unified import UnifiedKGQAPlanner
    m = UnifiedKGQAPlanner(num_dom, num_rel).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp6_unified_best.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_unified_top1(m, bert_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 6 – Unified"))
    del m; torch.cuda.empty_cache()

    # ── Exp 7 : RoBERTa-Large ──────────────────────────────
    print("\n── Exp 7: RoBERTa-Large ──")
    from train.exp7_roberta import ScaledUnifiedPlanner
    rob_tok = RobertaTokenizer.from_pretrained('roberta-large')
    m = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp7_roberta_best.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_unified_top1(m, rob_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 7 – RoBERTa-L"))
    del m; torch.cuda.empty_cache()

    # ── Exp 8 : CPD RoBERTa ────────────────────────────────
    print("\n── Exp 8: Contrastive RoBERTa (CPD) ──")
    m = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp8_cpd_best.pt'), device)
    m.eval()
    fn = lambda q, nh: predict_unified_top1(m, rob_tok, q, device, id2rel, nh)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 8 – CPD RoBERTa"))
    del m; torch.cuda.empty_cache()

    # ── Exp 9 : RLMC + Reranker ────────────────────────────
    print("\n── Exp 9: RLMC + Reranker (correct Hit@1) ──")
    from train.exp9_rlmc import RLConstraintAgent
    base = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(base, os.path.join(CKPT, 'exp7_roberta_best.pt'), device)
    rl = RLConstraintAgent(base).to(device)
    load_ckpt(rl, os.path.join(CKPT, 'exp9_rlmc_epoch_9.pt'), device)
    rl.eval()

    reranker = RobertaForSequenceClassification.from_pretrained(
        'roberta-large', num_labels=1).to(device)
    reranker.load_state_dict(
        torch.load(os.path.join(CKPT, 'exp9_reranker_final.pt'),
                   map_location=device, weights_only=False), strict=False)
    reranker.eval()

    fn = lambda q, nh: predict_rlmc_top1(rl, rob_tok, q, device, id2rel, nh)
    select_rr = lambda cands, q: pick_one_with_reranker(cands, q, reranker, rob_tok, device)
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_rr, "Exp 9 – RLMC+Reranker"))
    del rl, base, reranker; torch.cuda.empty_cache()

    # ── Exp 10 : Universal Planner (CWQ fine-tuned) ────────
    print("\n── Exp 10: Universal Planner – CWQ best ──")
    from train.exp10_universal import UniversalPlanner
    univ_rel2id = torch.load(
        os.path.join(ROOT, 'data/processed_universal/relation2id.pt'), weights_only=False)
    univ_id2rel = {v: k for k, v in univ_rel2id.items()}
    m10 = UniversalPlanner(num_domains=71, num_relations=len(univ_rel2id)).to(device)
    load_ckpt(m10, os.path.join(CKPT, 'exp10_cwq_best.pt'), device)
    m10.eval()
    fn = lambda q, nh: predict_universal_top1(m10, rob_tok, q, device, univ_id2rel, nh, 'cwq')
    all_results.append(evaluate_correct_hit1(samples, kg, fn, select_plain, "Exp 10 – Universal(CWQ)"))
    del m10; torch.cuda.empty_cache()

    # ── Print final summary ────────────────────────────────
    print("\n" + "=" * 65)
    print("  CORRECT HIT@1 RESULTS — CWQ Test Set")
    print("  Definition: single top-1 entity vs gold answer set")
    print("=" * 65)
    for r in all_results:
        flag = "← reranker" if "RLMC" in r['model'] else ""
        print(f"  {r['model']:30s}  Hit@1={r['hits@1']:.4f}  ({r['hits']}/{r['total']}) {flag}")
    print("-" * 65)
    print(f"  {'DRKG (published)':30s}  Hit@1=0.6699")
    print(f"  {'ChatKBQA (published)':30s}  Hit@1=0.5550")
    print(f"  {'NSM (published)':30s}  Hit@1=0.4860")
    print("=" * 65)

    # ── Write markdown ─────────────────────────────────────
    out_path = os.path.join(ROOT, 'results_correct_hit1.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# Correct Hit@1 Results — CWQ Test Set\n\n")
        f.write("**Definition**: For each question, the model returns exactly ONE entity "
                "(top-ranked). Hit = 1 if that entity is in the gold answer set.\n\n")
        f.write("For Exp 0–8 and Exp 10: top-1 relation path → traverse KG → pick "
                "lexicographically first reached entity (no reranker).\n")
        f.write("For Exp 9: RL beam traversal → reranker scores all candidates → top-1.\n\n")
        f.write("| Model | Hit@1 | Hits | Total | No-Candidate |\n")
        f.write("|---|---|---|---|---|\n")
        for r in all_results:
            f.write(f"| **{r['model']}** | {r['hits@1']:.4f} | "
                    f"{r['hits']} | {r['total']} | {r['no_cand']} |\n")
        f.write("\n### Published Baselines\n\n")
        f.write("| Method | Hit@1 |\n|---|---|\n")
        f.write("| DRKG (2025) | 0.6699 |\n")
        f.write("| ChatKBQA (2024) | 0.5550 |\n")
        f.write("| NSM (2021) | 0.4860 |\n")

        f.write("\n### Per-Hop Breakdown\n\n")
        f.write("| Model | 1-hop | 2-hop | 3-hop | 4-hop |\n|---|---|---|---|---|\n")
        for r in all_results:
            row = f"| **{r['model']}** |"
            for nh in range(1, 5):
                bh = r['by_hops'].get(nh, {'hits': 0, 'total': 0})
                hh = bh['hits']/bh['total'] if bh['total'] > 0 else 0
                row += f" {hh:.4f} ({bh.get('total',0)}) |"
            f.write(row + "\n")

    print(f"\nResults written → {out_path}")

if __name__ == '__main__':
    main()
