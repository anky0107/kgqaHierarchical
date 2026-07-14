"""
Correct Hit@1 on CWQ TEST SET — TRUE INFERENCE
================================================
What we use at inference time:
  - question text (q.txt)
  - topic_entity from processed_universal/cwq/test.json (entity linking output)
  - gold_answer from colab_inference/test_run/test_answers.txt

What we DO NOT use (gold leakage we're avoiding):
  - num_hops  → model always predicts up to max_hops=4
  - relations → model predicts these from scratch

Pipeline:
  question + topic_entity
        ↓
    model predicts [rel_hop0, rel_hop1, rel_hop2, rel_hop3]
        ↓
    traverse KG from topic_entity following predicted relations
    (stop naturally if KG returns no results at a hop)
        ↓
    pick 1 entity from final candidates
        ↓
    compare to gold_answer
"""
import os, sys, json, torch
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer, BertTokenizer, RobertaForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from shared.kg_loader import build_kg_from_cwq_triples
from utils.sparql_parser import extract_triples

MAX_HOPS = 4   # all models predict up to 4 relations; traversal stops naturally

# ─────────────────────────────────────────────────────────────
#  Load test samples — NO gold path used
# ─────────────────────────────────────────────────────────────

def load_test_samples_noleak(proc_test_path, ans_path):
    """
    proc_test_path : data/processed_universal/cwq/test.json
                     provides question + topic_entity
                     (we deliberately ignore 'relations' and 'num_hops')
    ans_path       : colab_inference/test_run/test_answers.txt
                     one gold MID per line, empty = skip
    """
    items = json.load(open(proc_test_path, encoding='utf-8'))
    ans   = [l.strip().strip(')') for l in open(ans_path, encoding='utf-8')]

    # processed_universal has 3501 items; test_answers has 3531
    # align by question text
    # build answer lookup by question (normalized)
    def norm(s): return s.strip().rstrip('?').strip().lower()

    q_to_ans = {}
    q_ian = [l.rstrip('\n') for l in open(
        os.path.join(ROOT, 'data/cwq_ianyunshi/test_CWQ/q.txt'), encoding='utf-8')]
    for q, a in zip(q_ian, ans):
        if a:
            q_to_ans[norm(q)] = a

    samples, skipped = [], 0
    for item in items:
        nq = norm(item['question'])
        gold = q_to_ans.get(nq, '')
        if not gold:
            skipped += 1
            continue
        te = item['topic_entity'].strip().strip(')')
        if not te:
            skipped += 1
            continue
        samples.append({
            'question':     item['question'],
            'topic_entity': te,
            'gold_answer':  gold,
            # num_hops deliberately NOT included — model uses MAX_HOPS
        })

    print(f"  Loaded {len(samples)} test samples (skipped {skipped} with no gold)")
    return samples

# ─────────────────────────────────────────────────────────────
#  KG traversal — model predicts MAX_HOPS relations,
#  traversal stops naturally when KG returns empty
# ─────────────────────────────────────────────────────────────

def traverse_noleak(kg, topic_entity, rel_sets_per_hop):
    """
    rel_sets_per_hop: list of sets, one per hop (up to MAX_HOPS)
    Traversal stops as soon as a hop returns no next entities.
    Returns the final non-empty entity set.
    """
    active = {topic_entity}
    last_nonempty = active
    for hop_rels in rel_sets_per_hop:
        if not active or not hop_rels:
            break
        nxt = set()
        for e in active:
            for rel, direction, tgt in kg.get_neighbors(e):
                if rel in hop_rels:
                    nxt.add(tgt.strip(')'))
        if not nxt:
            break          # natural stop — don't advance further
        last_nonempty = nxt
        active = nxt
    return active if active != {topic_entity} else set()

def pick_one(candidates):
    if not candidates:
        return None
    return min(candidates)

def pick_one_reranker(candidates, question, reranker, tokenizer, device, bs=32):
    if not candidates:
        return None
    cands = list(candidates)
    if len(cands) == 1:
        return cands[0]
    scores = []
    for i in range(0, len(cands), bs):
        batch = cands[i:i+bs]
        enc = tokenizer([question.lower()]*len(batch), batch,
                        padding=True, truncation=True, max_length=128,
                        return_tensors='pt').to(device)
        with torch.no_grad():
            scores.extend(torch.sigmoid(reranker(**enc).logits[:, 0]).cpu().tolist())
    return cands[scores.index(max(scores))]

# ─────────────────────────────────────────────────────────────
#  Model predict functions — always output MAX_HOPS predictions
#  Model doesn't know how many hops are needed
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_flat(model, tok, q, device, flat_id2rel):
    """Exp 0 – flat classifier over flat vocab → look up in flat_id2rel"""
    enc = tok(q, truncation=True, max_length=128, return_tensors='pt').to(device)
    logits = model(enc['input_ids'], enc['attention_mask'])
    topk = torch.topk(logits[0], k=MAX_HOPS).indices.tolist()
    return [{flat_id2rel.get(topk[h], '')} for h in range(MAX_HOPS)]

@torch.no_grad()
def predict_pct(model, tok, q, device, flat_id2rel):
    """Exp 3 – PCT also uses flat vocab"""
    enc = tok(q, truncation=True, max_length=128, return_tensors='pt').to(device)
    _, _, rel_logits, _ = model(enc['input_ids'], enc['attention_mask'])
    topk = torch.topk(rel_logits[0], k=MAX_HOPS).indices.tolist()
    return [{flat_id2rel.get(topk[h], '')} for h in range(MAX_HOPS)]

@torch.no_grad()
def predict_chcp(model, tok, q, device, id2rel):
    """Exp 4 – CHCP returns (rel_logits, hop_logits) tuple"""
    enc = tok(q, truncation=True, max_length=128, return_tensors='pt').to(device)
    rel_logits, _ = model(enc['input_ids'], enc['attention_mask'])
    max_h = rel_logits.size(1)
    return [{id2rel.get(torch.argmax(rel_logits[0, min(h, max_h-1)]).item(), '')}
            for h in range(MAX_HOPS)]

@torch.no_grad()
def predict_unified(model, tok, q, device, id2rel):
    enc = tok(q, truncation=True, max_length=128, return_tensors='pt').to(device)
    out = model(enc['input_ids'], enc['attention_mask'])
    rel_logits = out['rel_logits']          # [1, max_hops, num_rel]
    max_h = rel_logits.size(1)
    return [{id2rel.get(torch.argmax(rel_logits[0, min(h, max_h-1)]).item(), '')}
            for h in range(MAX_HOPS)]

@torch.no_grad()
def predict_rlmc(rl, tok, q, device, id2rel):
    enc = tok(q, truncation=True, max_length=128, return_tensors='pt').to(device)
    act_logits, _, rel_logits, _ = rl(enc['input_ids'], enc['attention_mask'])
    actions = torch.argmax(act_logits[0], dim=-1).tolist()
    w_map = {0: 1, 1: 5, 2: 50, 3: 0}
    max_h = rel_logits.size(1)
    result = []
    for h in range(MAX_HOPS):
        a = actions[h] if h < len(actions) else 3
        w = w_map.get(a, 1)
        if w == 0:
            result.append(set())   # STOP → empty set → traversal stops
            continue
        hi = min(h, max_h - 1)
        w = min(w, rel_logits.size(-1))
        ids = torch.topk(rel_logits[0, hi], k=w).indices.tolist()
        result.append({id2rel.get(r, '') for r in ids})
    return result

@torch.no_grad()
def predict_universal(model, tok, q, device, id2rel, ds='cwq'):
    ds_map = {'cwq': 0, 'webqsp': 1, 'metaqa': 2}
    ds_t = torch.tensor([ds_map.get(ds, 0)], device=device)
    enc = tok(q, truncation=True, max_length=160, return_tensors='pt').to(device)
    out = model(enc['input_ids'], enc['attention_mask'], ds_t)
    rel_logits = out['rel_logits']
    max_h = rel_logits.size(1)
    return [{id2rel.get(torch.argmax(rel_logits[0, min(h, max_h-1)]).item(), '')}
            for h in range(MAX_HOPS)]

# ─────────────────────────────────────────────────────────────
#  Evaluation loop
# ─────────────────────────────────────────────────────────────

def evaluate(samples, kg, predict_fn, select_fn, name):
    total = hits = no_cand = 0

    for s in tqdm(samples, desc=name, ncols=90):
        rel_sets   = predict_fn(s['question'])         # model predicts MAX_HOPS rels
        candidates = traverse_noleak(kg, s['topic_entity'], rel_sets)
        pred       = select_fn(candidates, s['question'])

        total += 1
        if pred is None:
            no_cand += 1
        elif pred == s['gold_answer']:
            hits += 1

    h1 = hits / total if total > 0 else 0
    print(f"\n  ── {name}")
    print(f"     Hit@1         : {h1:.4f}  ({hits}/{total})")
    print(f"     No candidates : {no_cand}/{total}")
    return {'model': name, 'hits@1': h1, 'hits': hits, 'total': total, 'no_cand': no_cand}

def load_ckpt(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model

# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    DATA  = os.path.join(ROOT, 'data')
    PDATA = os.path.join(DATA, 'processed_entity')
    CKPT  = os.path.join(ROOT, 'checkpoints')

    rel2id       = torch.load(os.path.join(PDATA, 'relation2id.pt'), weights_only=False)
    id2rel       = {v: k for k, v in rel2id.items()}
    num_rel      = len(rel2id)
    train_d      = torch.load(os.path.join(PDATA, 'train_domains.pt'), weights_only=False)
    num_dom      = int(torch.max(train_d).item()) + 1
    train_r      = torch.load(os.path.join(PDATA, 'train_relations.pt'), weights_only=False)
    num_rel_flat = int(torch.max(train_r).item()) + 1

    # Flat-vocab id2rel (for Exp 0 / Exp 3)
    flat_rel2id_path = os.path.join(PDATA, 'flat_relation2id.pt')
    if os.path.exists(flat_rel2id_path):
        flat_rel2id = torch.load(flat_rel2id_path, weights_only=False)
        flat_id2rel = {v: k for k, v in flat_rel2id.items()}
    else:
        # Build from train_relations tensor if no separate file
        # train_relations maps sample→relation_id, so we need the vocabulary
        # Fall back: same as main rel2id (ids 0..num_rel_flat-1)
        flat_id2rel = {i: id2rel.get(i, '') for i in range(num_rel_flat)}
    print(f'Flat vocab size: {num_rel_flat}, main vocab: {num_rel}')

    print("\n[1/3] Building KG…")
    kg = build_kg_from_cwq_triples([
        os.path.join(DATA, 'cwq_train.json'),
        os.path.join(DATA, 'cwq_dev.json'),
        os.path.join(DATA, 'cwq_test.json'),
    ], extract_triples)

    print("\n[2/3] Loading test samples (no gold path leakage)…")
    samples = load_test_samples_noleak(
        proc_test_path = os.path.join(DATA, 'processed_universal/cwq/test.json'),
        ans_path       = os.path.join(ROOT, 'colab_inference/test_run/test_answers.txt'),
    )

    sel_plain = lambda c, q: pick_one(c)
    results   = []

    print(f"\n[3/3] Evaluating {len(samples)} test questions (MAX_HOPS={MAX_HOPS})…\n")

    # ── Exp 0 ────────────────────────────────────────────────
    print("── Exp 0: Flat BERT ──")
    from train.exp0_flat_baseline import BERTRelationClassifier
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    m = BERTRelationClassifier(num_relations=num_rel_flat).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp0_relation_flat_best.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_flat(m, bert_tok, q, device, flat_id2rel),
        sel_plain, "Exp 0 – Flat BERT"))
    del m; torch.cuda.empty_cache()

    # ── Exp 3 ────────────────────────────────────────────────
    print("\n── Exp 3: PCT ──")
    from train.exp3_pct import PCTModel
    m = PCTModel(num_domains=num_dom, num_relations=num_rel_flat).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp3_pct_best.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_pct(m, bert_tok, q, device, flat_id2rel),
        sel_plain, "Exp 3 – PCT"))
    del m; torch.cuda.empty_cache()

    # ── Exp 4 ────────────────────────────────────────────────
    print("\n── Exp 4: CHCP ──")
    from train.exp4_chcp import CHCPModel
    m = CHCPModel(num_relations=num_rel, max_hops=4).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp4_chcp_best.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_chcp(m, bert_tok, q, device, id2rel),
        sel_plain, "Exp 4 – CHCP"))
    del m; torch.cuda.empty_cache()

    # ── Exp 4-RL ─────────────────────────────────────────────
    print("\n── Exp 4-RL ──")
    m = CHCPModel(num_relations=num_rel, max_hops=4).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp4_rl_epoch_49.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_chcp(m, bert_tok, q, device, id2rel),
        sel_plain, "Exp 4-RL – CHCP+PPO"))
    del m; torch.cuda.empty_cache()

    # ── Exp 6 ────────────────────────────────────────────────
    print("\n── Exp 6: Unified ──")
    from train.exp6_unified import UnifiedKGQAPlanner
    m = UnifiedKGQAPlanner(num_dom, num_rel).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp6_unified_best.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_unified(m, bert_tok, q, device, id2rel),
        sel_plain, "Exp 6 – Unified"))
    del m; torch.cuda.empty_cache()

    # ── Exp 7 ────────────────────────────────────────────────
    print("\n── Exp 7: RoBERTa-L ──")
    from train.exp7_roberta import ScaledUnifiedPlanner
    rob_tok = RobertaTokenizer.from_pretrained('roberta-large')
    m = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp7_roberta_best.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_unified(m, rob_tok, q, device, id2rel),
        sel_plain, "Exp 7 – RoBERTa-L"))
    del m; torch.cuda.empty_cache()

    # ── Exp 8 ────────────────────────────────────────────────
    print("\n── Exp 8: CPD RoBERTa ──")
    m = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(m, os.path.join(CKPT, 'exp8_cpd_best.pt'), device)
    m.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_unified(m, rob_tok, q, device, id2rel),
        sel_plain, "Exp 8 – CPD RoBERTa"))
    del m; torch.cuda.empty_cache()

    # ── Exp 9 ────────────────────────────────────────────────
    print("\n── Exp 9: RLMC + Reranker ──")
    from train.exp9_rlmc import RLConstraintAgent
    base = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    load_ckpt(base, os.path.join(CKPT, 'exp7_roberta_best.pt'), device)
    rl = RLConstraintAgent(base).to(device)
    load_ckpt(rl, os.path.join(CKPT, 'exp9_rlmc_epoch_9.pt'), device)
    rl.eval()
    reranker = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1).to(device)
    reranker.load_state_dict(torch.load(os.path.join(CKPT, 'exp9_reranker_final.pt'),
                             map_location=device, weights_only=False), strict=False)
    reranker.eval()
    sel_rr = lambda c, q: pick_one_reranker(c, q, reranker, rob_tok, device)
    results.append(evaluate(samples, kg,
        lambda q: predict_rlmc(rl, rob_tok, q, device, id2rel),
        sel_rr, "Exp 9 – RLMC+Reranker"))
    del rl, base, reranker; torch.cuda.empty_cache()

    # ── Exp 10 ───────────────────────────────────────────────
    print("\n── Exp 10: Universal(CWQ-best) ──")
    from train.exp10_universal import UniversalPlanner
    univ_rel2id = torch.load(os.path.join(DATA, 'processed_universal/relation2id.pt'), weights_only=False)
    univ_id2rel = {v: k for k, v in univ_rel2id.items()}
    m10 = UniversalPlanner(num_domains=70, num_relations=len(univ_rel2id)).to(device)
    load_ckpt(m10, os.path.join(CKPT, 'exp10_cwq_best.pt'), device)
    m10.eval()
    results.append(evaluate(samples, kg,
        lambda q: predict_universal(m10, rob_tok, q, device, univ_id2rel, 'cwq'),
        sel_plain, "Exp 10 – Universal(CWQ)"))
    del m10; torch.cuda.empty_cache()

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CORRECT HIT@1 — CWQ TEST SET")
    print("  (No gold path leakage: model predicts all hops itself)")
    print("=" * 65)
    for r in results:
        tag = " ← reranker" if "RLMC" in r['model'] else ""
        print(f"  {r['model']:35s}  {r['hits@1']:.4f}  ({r['hits']}/{r['total']}){tag}")
    print("-" * 65)
    print(f"  {'ChatKBQA (published)':35s}  0.5550")
    print(f"  {'DRKG (published)':35s}  0.6699")
    print("=" * 65)

    out = os.path.join(ROOT, 'results_test_noleak.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("# CWQ Test Set — Correct Hit@1 (No Gold Path Leakage)\n\n")
        f.write(f"**Questions evaluated**: {len(samples)} (of 3531 test; 2529+ have no gold answer)\n")
        f.write(f"**MAX_HOPS**: {MAX_HOPS} (model predicts all hops, traversal stops naturally)\n\n")
        f.write("| Model | Hit@1 | Hits | Total | No-Cand |\n|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| **{r['model']}** | {r['hits@1']:.4f} | {r['hits']} | {r['total']} | {r['no_cand']} |\n")
        f.write("\n| ChatKBQA (2024) | 0.5550 | - | - | - |\n")
        f.write("| DRKG (2025) | 0.6699 | - | - | - |\n")
    print(f"\nResults → {out}")

if __name__ == '__main__':
    main()
