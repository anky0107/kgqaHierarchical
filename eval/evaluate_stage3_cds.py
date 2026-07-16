import os, sys, json, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# evaluate_stage3_cds.py  --  Stage 3 CDS Full Entity-Level Evaluation
# =============================================================================
#
# SUMMARY
# -------
# Evaluates the full Cascading Dual-Stage Filtering (CDS) pipeline missing
# from evaluate_e2e.py. evaluate_e2e.py only measures relation-path accuracy
# (path-match Hit@1) and never returns an entity answer. This script measures
# the actual end-to-end metric: entity-level Hit@1 on the CWQ dev set.
#
# PIPELINE STAGES
# ---------------
#   F1  (pre-collected, on disk as data/exp16_cds_dev.json)
#       Raw candidate entity pool from STRL agent (Exp-15) graph traversal.
#       Records where F1 never reached a gold entity count as HARD MISSES
#       and remain in the denominator (dropping them inflates scores ~28%).
#
#   F2  PathAwareRanker  (all-mpnet-base-v2 + 3-input MLP fusion head)
#       Checkpoint : checkpoints/exp25_s2_listwise.pt (fallback for exp27)
#       Re-ranks candidate pool scoring (question, path, entity) triples.
#       Keeps top-50 for F3.
#
#   F3  Flan-T5-base Generative Judge  (SFT and DPO variants)
#       SFT checkpoint : checkpoints/exp31_t5_mc_s3.pt
#       DPO checkpoint : checkpoints/exp38_t5_dpo_s3.pt  [paper result]
#       Listwise MC prompt -> generates exact entity name as free text.
#
# DENOMINATOR: ALL 3502 DEV RECORDS ALWAYS COUNT
# -----------------------------------------------
# 3502 total CWQ dev questions. All 3502 are in the denominator for every
# metric. 1000 records where F1 missed are automatic misses (Hit=0).
# Dropping them would inflate scores by ~28%.
#
# METRICS REPORTED (5 stages)
# ---------------------------
#   1. F1-recall Hit@1  -- gold anywhere in raw F1 pool? (upper bound)
#   2. F1-first  Hit@1  -- first F1 candidate verbatim, no ranking (baseline)
#   3. F2-only   Hit@1  -- top-1 after MPNet re-ranking (no generative F3)
#   4. F1->F2->F3-SFT   -- full pipeline with SFT judge
#   5. F1->F2->F3-DPO   -- full pipeline with DPO judge  [paper result]
#
# F2 vs F3 AGREEMENT ANALYSIS
# ----------------------------
# Per-record tracking: F3 agrees with F2 / F3 better / F3 worse.
# Explains why F2 and F3 can show identical aggregate Hit@1: F3 may flip
# some correct->wrong AND wrong->correct, cancelling in the aggregate.
#
# FUZZY ENTITY MATCH
# ------------------
#   pred == gold  (exact, lowercased)
#   gold in pred  (verbose T5: 'The answer is Paris' matches 'Paris')
#   pred in gold  (abbreviation: 'US' matches 'United States')
#
# USAGE
# -----
#   python paper_code/eval/evaluate_stage3_cds.py              # full 3502
#   python paper_code/eval/evaluate_stage3_cds.py --limit 200  # quick test
#   python eval/evaluate_stage3_cds.py                         # same script
# =============================================================================

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Auto-detect project root: walk up until dir has both data/ and checkpoints/
_here = os.path.dirname(os.path.abspath(__file__))
ROOT = _here
for _ in range(4):
    if (os.path.isdir(os.path.join(ROOT, 'data')) and
            os.path.isdir(os.path.join(ROOT, 'checkpoints'))):
        break
    ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer

CKPT_DIR         = os.path.join(ROOT, 'checkpoints')
F2_CKPT_PRIMARY  = os.path.join(CKPT_DIR, 'exp27_s2_mpnet.pt')
F2_CKPT_FALLBACK = os.path.join(CKPT_DIR, 'exp25_s2_listwise.pt')
F3_DPO_CKPT      = os.path.join(CKPT_DIR, 'exp38_t5_dpo_s3.pt')
F3_SFT_CKPT      = os.path.join(CKPT_DIR, 'exp31_t5_mc_s3.pt')
CDS_DEV_JSON     = os.path.join(ROOT, 'data', 'exp16_cds_dev.json')
F2_ENCODER       = 'sentence-transformers/all-mpnet-base-v2'
F3_MODEL         = 'google/flan-t5-base'


class PathAwareRanker(nn.Module):
    # F2 ranker -- verbatim copy from stage3_cds/train_f2_path_ranker.py
    # Architecture: shared MPNet encoder -> CLS for (question, path, entity)
    #               -> MLP([q; p; e]) -> scalar relevance score
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(F2_ENCODER)
        h = self.encoder.config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(h * 3, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, q_ids, q_mask, p_ids, p_mask, e_ids, e_mask):
        enc = self.encoder
        q = enc(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p = enc(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e = enc(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q, p, e], dim=-1)).squeeze(-1)


def path_to_nl(path):
    # Convert nested-list / flat-list / str KG path to readable string.
    if isinstance(path, str):
        return path.replace('.', ' ').replace('_', ' ')
    if isinstance(path, list):
        parts = []
        for hop in path:
            if isinstance(hop, list):
                if hop:
                    parts.append(hop[0].replace('.', ' ').replace('_', ' '))
            else:
                parts.append(str(hop).replace('.', ' ').replace('_', ' '))
        return ' -> '.join(parts)
    return str(path)


def build_f3_prompt(question, candidates, path):
    # Listwise MC prompt for Flan-T5 judge. Matches train_f3_sft.py format.
    path_nl = path_to_nl(path)
    prompt = 'Question: {}\n\nCandidates:\n'.format(question)
    for i, c in enumerate(candidates, 1):
        name = c.get('name', '').strip() or '[UNK]'
        if path_nl:
            prompt += '{}. {} (Path: {})\n'.format(i, name, path_nl)
        else:
            prompt += '{}. {}\n'.format(i, name)
    prompt += '\nWhich of the above candidates is the correct answer to the question? Answer with the exact name.'
    return prompt


def is_hit(pred, gold_names):
    # Fuzzy entity match: exact | gold in pred (verbose T5) | pred in gold (abbrev)
    pred_l = pred.lower().strip()
    for g in gold_names:
        g_l = g.lower().strip()
        if not g_l:
            continue
        if pred_l == g_l or g_l in pred_l or pred_l in g_l:
            return True
    return False


@torch.no_grad()
def f1_filter_candidates(s1_model, s1_tok, question, candidates, all_embs, mid2idx, device, top_k=200):
    # F1 Fast Filter -- prunes massive candidate pools down to top_k using pre-computed CPU embeddings
    if len(candidates) <= top_k:
        return candidates
        
    q_enc = s1_tok(question, return_tensors='pt', padding=True, truncation=True).to(device)
    q_emb = s1_model(**q_enc).last_hidden_state[:, 0, :].cpu()
    
    cand_idx = []
    valid_cands = []
    for c in candidates:
        mid = c.get('mid')
        if mid and mid in mid2idx:
            cand_idx.append(mid2idx[mid])
            valid_cands.append(c)
            
    if not valid_cands:
        return candidates[:top_k]
        
    e_embs = all_embs[cand_idx]
    sims = F.cosine_similarity(q_emb, e_embs)
    
    k = min(top_k, len(valid_cands))
    top_idx = torch.topk(sims, k).indices.tolist()
    
    return [valid_cands[i] for i in top_idx]


@torch.no_grad()
def f2_rank_candidates(model, tok, question, path, candidates, device, top_k=50, chunk=32):
    # Score all candidates and return top_k sorted descending by score.
    if not candidates:
        return []
    path_str = path_to_nl(path)
    names = [str(c.get('name', '')) for c in candidates]
    qs = [question] * len(candidates)
    ps = [path_str] * len(candidates)
    all_sc = []
    for i in range(0, len(candidates), chunk):
        qe = tok(qs[i:i+chunk],    padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        pe = tok(ps[i:i+chunk],    padding=True, truncation=True, max_length=64,  return_tensors='pt').to(device)
        ee = tok(names[i:i+chunk], padding=True, truncation=True, max_length=64,  return_tensors='pt').to(device)
        sc = model(qe['input_ids'], qe['attention_mask'],
                   pe['input_ids'], pe['attention_mask'],
                   ee['input_ids'], ee['attention_mask'])
        all_sc.append(sc.cpu())
    all_sc_t = torch.cat(all_sc, dim=0)
    ranked_idx = torch.argsort(all_sc_t, descending=True).tolist()
    return [candidates[i] for i in ranked_idx[:top_k]]


@torch.no_grad()
def f3_generate(model, tok, prompt, device, max_new_tokens=64):
    # Generate entity name with beam search (beams=4). Returns decoded string.
    enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def evaluate_cds(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n' + '='*65)
    print('  Stage 3 CDS -- Full Entity-Level Evaluation')
    print('  device = {}'.format(device))
    print('='*65 + '\n')

    print('Loading: {}'.format(CDS_DEV_JSON))
    with open(CDS_DEV_JSON, encoding='utf-8') as f:
        data = json.load(f)
    n_with_gold = sum(1 for r in data
                      if any(c.get('is_gold') for c in r.get('candidates', [])))
    print('  Total records     : {}  (all in denominator)'.format(len(data)))
    print('  STRL retrieved gold: {}'.format(n_with_gold))
    print('  STRL missed       : {}  (auto-miss, still in denom)'.format(
          len(data) - n_with_gold))

    if args.limit > 0:
        data = data[:args.limit]
        print('  [--limit] Using first {} records.'.format(len(data)))
    print()

    print('Loading F1 Fast Filter embeddings to CPU RAM (this takes a moment)...')
    import torch.nn.functional as F
    emb_data = torch.load(os.path.join(ROOT, 'data', 'exp16_entity_embs.pt'), map_location='cpu', weights_only=False)
    all_mids = emb_data['mids']
    mid2idx  = {mid: i for i, mid in enumerate(all_mids)}
    all_embs = emb_data['embs']
    del emb_data
    
    print('Loading F1 Bi-Encoder: exp16_s1_bi.pt')
    f1_tok   = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    f1_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    f1_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints', 'exp16_s1_bi.pt'), map_location=device, weights_only=False))
    f1_model.eval()
    print('  F1 ready')

    f2_ckpt = F2_CKPT_PRIMARY if os.path.isfile(F2_CKPT_PRIMARY) else F2_CKPT_FALLBACK
    print('Loading F2: {}'.format(os.path.basename(f2_ckpt)))
    f2_tok   = AutoTokenizer.from_pretrained(F2_ENCODER)
    f2_model = PathAwareRanker().to(device)
    f2_model.load_state_dict(torch.load(f2_ckpt, map_location=device, weights_only=False))
    f2_model.eval()
    print('  F2 ready')

    print('Loading F3-DPO: {}'.format(os.path.basename(F3_DPO_CKPT)))
    f3_tok = T5Tokenizer.from_pretrained(F3_MODEL)
    f3_dpo = T5ForConditionalGeneration.from_pretrained(F3_MODEL).to(device)
    f3_dpo.load_state_dict(torch.load(F3_DPO_CKPT, map_location=device, weights_only=False))
    f3_dpo.eval()
    print('  F3-DPO ready')

    print('Loading F3-SFT: {}'.format(os.path.basename(F3_SFT_CKPT)))
    f3_sft = T5ForConditionalGeneration.from_pretrained(F3_MODEL).to(device)
    f3_sft.load_state_dict(torch.load(F3_SFT_CKPT, map_location=device, weights_only=False))
    f3_sft.eval()
    print('  F3-SFT ready')
    print('=' * 65 + '\n')

    total = 0
    hits_f1_recall = hits_f1_first = hits_f2 = hits_sft = hits_dpo = 0
    f3_sft_agree  = f3_sft_better  = f3_sft_worse  = 0
    f3_dpo_agree  = f3_dpo_better  = f3_dpo_worse  = 0
    by_depth = defaultdict(lambda: {
        'total': 0, 'f1_recall': 0, 'f1_first': 0, 'f2': 0, 'sft': 0, 'dpo': 0})

    for record in tqdm(data, desc='CDS Eval (F1->F2->F3)', ncols=80):
        question   = record['question']
        path       = record['path']
        candidates = record['candidates']
        depth      = len(path) if isinstance(path, list) else 1
        gold_names = [c.get('name', '') for c in candidates if c.get('is_gold')]

        total += 1
        by_depth[depth]['total'] += 1

        if gold_names:
            hits_f1_recall += 1
            by_depth[depth]['f1_recall'] += 1

        first_name = candidates[0].get('name', '') if candidates else ''
        if gold_names and is_hit(first_name, gold_names):
            hits_f1_first += 1
            by_depth[depth]['f1_first'] += 1

        if not gold_names:
            continue
            
        # --- STAGE 3, F1: FAST FILTER (BI-ENCODER) ---
        f1_cands = f1_filter_candidates(f1_model, f1_tok, question, candidates, all_embs, mid2idx, device, top_k=200)

        # --- STAGE 3, F2: PATH RANKER (MPNET) ---
        ranked  = f2_rank_candidates(f2_model, f2_tok, question, path,
                                      f1_cands, device, top_k=args.f2_top_k)
        top1_f2 = ranked[0].get('name', '') if ranked else ''
        f2_hit  = is_hit(top1_f2, gold_names)
        if f2_hit:
            hits_f2 += 1
            by_depth[depth]['f2'] += 1

        f3_cands = ranked[:args.max_cands]
        prompt   = build_f3_prompt(question, f3_cands, path)

        pred_sft = f3_generate(f3_sft, f3_tok, prompt, device)
        sft_hit  = is_hit(pred_sft, gold_names)
        if sft_hit:
            hits_sft += 1
            by_depth[depth]['sft'] += 1
        if   sft_hit == f2_hit:       f3_sft_agree  += 1
        elif sft_hit and not f2_hit:  f3_sft_better += 1
        else:                         f3_sft_worse  += 1

        pred_dpo = f3_generate(f3_dpo, f3_tok, prompt, device)
        dpo_hit  = is_hit(pred_dpo, gold_names)
        if dpo_hit:
            hits_dpo += 1
            by_depth[depth]['dpo'] += 1
        if   dpo_hit == f2_hit:       f3_dpo_agree  += 1
        elif dpo_hit and not f2_hit:  f3_dpo_better += 1
        else:                         f3_dpo_worse  += 1

    def pct(n):
        return n / total if total else 0.0

    print('\n' + '='*65)
    print('  STAGE 3 CDS -- ENTITY-LEVEL HIT@1  (denom = {} records)'.format(total))
    print('='*65)
    print('  F2 ckpt : {}  |  top-k: {}'.format(os.path.basename(f2_ckpt), args.f2_top_k))
    print('  F3-DPO  : {}  |  max_cands: {}'.format(os.path.basename(F3_DPO_CKPT), args.max_cands))
    print('  F3-SFT  : {}'.format(os.path.basename(F3_SFT_CKPT)))
    print('-'*65)
    print('  {:38} {:>7}  {:>6} / {}'.format('Stage', 'Hit@1', 'Hits', total))
    print('-'*65)
    for label, h in [
        ('[1] F1-recall  (gold in pool, ceiling)',    hits_f1_recall),
        ('[2] F1-first   (no ranking, no-CDS base)',  hits_f1_first),
        ('[3] F2-only    (MPNet re-rank, no gen.)',    hits_f2),
        ('[4] F1->F2->F3-SFT',                         hits_sft),
        ('[5] F1->F2->F3-DPO  [full CDS]',             hits_dpo),
    ]:
        print('  {:38} {:>7.4f}  {:>6}'.format(label, pct(h), h))
    print('='*65)

    print('\n  F2 vs F3 agreement  (over {} records where F1 found gold)'.format(
          hits_f1_recall))
    print('  {:6} {:>8}  {:>10}  {:>9}'.format('', 'Agree', 'F3 Better', 'F3 Worse'))
    print('  SFT    {:>8}  {:>10}  {:>9}'.format(
          f3_sft_agree, f3_sft_better, f3_sft_worse))
    print('  DPO    {:>8}  {:>10}  {:>9}'.format(
          f3_dpo_agree, f3_dpo_better, f3_dpo_worse))

    print('\n  Per-depth breakdown:\n')
    print('  {:6} {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
          'Depth', 'N', 'F1-rec', 'F1-1st', 'F2@1', 'SFT@1', 'DPO@1'))
    print('  ' + '-'*58)
    for d in sorted(by_depth):
        row = by_depth[d]
        t   = row['total']
        if t == 0:
            continue
        print('  {}-hop  {:>5}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}'.format(
              d, t,
              row['f1_recall'] / t, row['f1_first'] / t,
              row['f2'] / t,        row['sft'] / t,        row['dpo'] / t))
    print()

    out = {
        'total_in_denominator': total,
        'f1_retrieved_gold'   : hits_f1_recall,
        'f1_missed'           : total - hits_f1_recall,
        'f2_ckpt'             : os.path.basename(f2_ckpt),
        'f3_dpo_ckpt'         : os.path.basename(F3_DPO_CKPT),
        'f3_sft_ckpt'         : os.path.basename(F3_SFT_CKPT),
        'config'              : {'f2_top_k': args.f2_top_k, 'f3_max_cands': args.max_cands},
        'hit1': {
            'f1_recall': pct(hits_f1_recall),
            'f1_first' : pct(hits_f1_first),
            'f2_only'  : pct(hits_f2),
            'f3_sft'   : pct(hits_sft),
            'f3_dpo'   : pct(hits_dpo),
        },
        'f2_vs_f3': {
            'sft': {'agree': f3_sft_agree, 'f3_better': f3_sft_better, 'f3_worse': f3_sft_worse},
            'dpo': {'agree': f3_dpo_agree, 'f3_better': f3_dpo_better, 'f3_worse': f3_dpo_worse},
        },
        'by_depth': {
            str(d): {
                'total'     : v['total'],
                'f1_recall' : v['f1_recall'] / v['total'] if v['total'] else 0,
                'f1_first'  : v['f1_first']  / v['total'] if v['total'] else 0,
                'f2'        : v['f2']         / v['total'] if v['total'] else 0,
                'sft'       : v['sft']        / v['total'] if v['total'] else 0,
                'dpo'       : v['dpo']        / v['total'] if v['total'] else 0,
            }
            for d, v in by_depth.items()
        },
    }
    out_path = os.path.join(ROOT, 'eval', 'stage3_cds_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(out, fout, indent=2)
    print('  Results saved -> {}\n'.format(out_path))


def parse_args():
    p = argparse.ArgumentParser(
        description='Stage 3 CDS entity-level Hit@1 (denom=full dev set)')
    p.add_argument('--max_cands', type=int, default=15)
    p.add_argument('--f2_top_k',  type=int, default=50)
    p.add_argument('--limit',     type=int, default=0, help='0 = full 3502 dev set')
    return p.parse_args()


if __name__ == '__main__':
    evaluate_cds(parse_args())