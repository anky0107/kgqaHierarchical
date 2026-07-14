"""
eval_dev.py — End-to-End Dev Set Evaluation
============================================

Runs the complete inference pipeline on the CWQ dev set and reports:
  - Stage I  Path Accuracy  (relation prediction quality)
  - Stage I+II Reasoning Recall  (is gold entity in candidate set?)
  - Stage I+II+III Hit@1  (is top-1 CDS output the gold entity?)

Supports both pipelines:
  Pipeline A: exp7 (frozen) + exp9 (A2C policy) + LMDB + CDS
  Pipeline B: exp15 (STRL, joint) + LMDB + CDS

Also supports CDS variants:
  CDS baseline:  exp16v2 Stage 1→2→3 (name only)
  CDS enriched:  exp22 subgraph context (entity + KG neighbours)
  CDS verified:  exp23 two-pass LMDB verification after Stage 3

HOW TO RUN
----------
  # Pipeline A (exp7 + exp9) + baseline CDS:
  python eval_dev.py --pipeline a

  # Pipeline B (exp15 STRL) + baseline CDS:
  python eval_dev.py --pipeline b

  # Pipeline A + subgraph enriched CDS (exp22):
  python eval_dev.py --pipeline a --cds enriched

  # Pipeline B + two-pass verification (exp23):
  python eval_dev.py --pipeline b --cds verified

  # All combinations in one run:
  python eval_dev.py --all

  # Only measure Reasoning Recall (no CDS):
  python eval_dev.py --pipeline a --recall_only

FLAGS
-----
  --pipeline    a | b           Which traversal pipeline (default: b)
  --cds         baseline | enriched | verified | none
                                Which CDS variant (default: baseline)
  --recall_only                 Stop after traversal, skip CDS
  --cwq_dev     path            CWQ dev JSON (default: data/cwq_dev.json)
  --kg_path     path            KG path (default: data/processed_kg/augmented_kg.pt)
  --exp7_ckpt   path            exp7 checkpoint
  --exp9_ckpt   path            exp9 checkpoint (best epoch)
  --exp15_ckpt  path            exp15 checkpoint
  --s1_ckpt     path            CDS Stage 1 checkpoint
  --s2_ckpt     path            CDS Stage 2 checkpoint
  --s3_ckpt     path            CDS Stage 3 checkpoint
  --max_questions int           Limit questions evaluated (default: all)
  --batch_size  int             Encoding batch size (default: 1)
  --top_n_subgraph int          Neighbours per entity in enriched mode (default: 5)
  --top_k_verify int            Candidates to verify in verified mode (default: 5)

OUTPUT
------
  metrics/eval_dev_{pipeline}_{cds}_{timestamp}.csv
  Prints a summary table to stdout.
"""

import os, sys, json, argparse, time
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification,
                          RobertaTokenizer)
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

# Import model classes directly without triggering exp6_unified's sparql_parser dependency.
# We only need the model class definitions, not the training dataset classes.
import importlib.util, types

def _load_module(name, filepath):
    """Load a module from an absolute path without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub out utils.sparql_parser so exp6_unified doesn't crash on import
if "utils.sparql_parser" not in sys.modules:
    stub = types.ModuleType("utils.sparql_parser")
    stub.find_reasoning_path = lambda *a, **kw: None
    sys.modules["utils.sparql_parser"] = stub
if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")

_exp7  = _load_module("train.exp7_roberta",  os.path.join(ROOT, "train/exp7_roberta.py"))
_exp9  = _load_module("train.exp9_rlmc",     os.path.join(ROOT, "train/exp9_rlmc.py"))
_exp15 = _load_module("train.exp15_strl",    os.path.join(ROOT, "train/exp15_strl.py"))
_exp16 = _load_module("train.exp16v2_train", os.path.join(ROOT, "train/exp16v2_train.py"))

ScaledUnifiedPlanner       = _exp7.ScaledUnifiedPlanner
RLConstraintAgent          = _exp9.RLConstraintAgent
STRLAgent                  = _exp15.STRLAgent
RelationEmbeddingBank      = _exp15.RelationEmbeddingBank
semantic_beam_with_kg_filter = _exp15.semantic_beam_with_kg_filter
PathAwareRanker            = _exp16.PathAwareRanker

# ── Action constants (same as exp9/exp15) ────────────────────────────────────
ACTION_TIGHT  = 0
ACTION_MEDIUM = 1
ACTION_LOOSE  = 2
ACTION_STOP   = 3
BEAM_SIZES    = {ACTION_TIGHT: 1, ACTION_MEDIUM: 5, ACTION_LOOSE: 50}


# ─────────────────────────────────────────────────────────────
#  Utility: relation NL conversion  (same heuristic as exp15)
# ─────────────────────────────────────────────────────────────

def _rel_to_nl(rel_id: str) -> str:
    parts = rel_id.split(".")
    if len(parts) >= 3:
        subject   = parts[-2].replace("_", " ")
        predicate = parts[-1].replace("_", " ")
        if (predicate.endswith("s") or "owned" in predicate
                or "founded" in predicate or "won" in predicate):
            return f"{subject} HAS {predicate}"
        return f"{subject} {predicate}"
    if len(parts) == 2:
        return parts[-1].replace("_", " ")
    return rel_id.replace(".", " ").replace("_", " ")


# ─────────────────────────────────────────────────────────────
#  Subgraph description for enriched CDS (exp22 logic)
# ─────────────────────────────────────────────────────────────

def build_subgraph_str(entity_mid: str, entity_name: str,
                        kg: dict, rel2id: dict, top_n: int = 5) -> str:
    if not entity_mid or not kg:
        return entity_name
    forward  = kg.get("forward",  {})
    backward = kg.get("backward", {})
    pairs = []
    for rel_str, target in forward.get(entity_mid, []):
        if rel_str in rel2id:
            pairs.append((rel_str, target))
    if len(pairs) < top_n:
        for rel_str, src in backward.get(entity_mid, []):
            if rel_str in rel2id:
                pairs.append((rel_str, src))
    if not pairs:
        return entity_name
    rel_targets = defaultdict(list)
    for rel_str, target in pairs:
        rel_targets[rel_str].append(target)
    sorted_rels = sorted(rel_targets.items(),
                         key=lambda x: len(x[1]), reverse=True)[:top_n]
    parts = [entity_name]
    for rel_str, targets in sorted_rels:
        shown = ", ".join(str(t) for t in targets[:3])
        parts.append(f"{_rel_to_nl(rel_str)}: {shown}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
#  Path existence check for verified CDS (exp23 logic)
# ─────────────────────────────────────────────────────────────

def path_exists(topic_mid: str, answer_mid: str,
                rel_seq: list, kg: dict) -> bool:
    if not rel_seq or not topic_mid or not answer_mid:
        return False
    forward = kg.get("forward", {})
    current = {topic_mid}
    for i, rel in enumerate(rel_seq):
        nxt = set()
        for mid in current:
            for edge_rel, target in forward.get(mid, []):
                if edge_rel == rel:
                    nxt.add(target)
        if not nxt:
            return False
        if i == len(rel_seq) - 1:
            return answer_mid in nxt
        current = nxt
    return False


# ─────────────────────────────────────────────────────────────
#  CWQ dev loader
# ─────────────────────────────────────────────────────────────

def load_cwq_dev(cwq_dev_path: str, root: str) -> list:
    """
    Load dev questions from the pre-computed STRLDataset cache.
    This cache was built by exp15_strl.py and contains:
      question, domain, path (rel_ids), num_hops, topic_entity, avail_rels

    We also need gold answer MIDs — these come from cwq_dev.json directly.
    The cache has relation IDs, so we also need id2rel to reconstruct strings.
    """
    import json

    # ── Option A: Load from dataset cache (fast, already processed) ──────────
    cache_path = os.path.join(root, "data/processed_entity/dataset_cache_dev.pt")
    if os.path.exists(cache_path):
        print(f"[EvalDev] Loading dev cache from {cache_path} ...")
        samples = torch.load(cache_path, map_location="cpu")
        print(f"[EvalDev] Cache loaded: {len(samples)} samples")
    else:
        # ── Option B: Fall back to raw CWQ JSON with sparql_parser ───────────
        print(f"[EvalDev] Cache not found, loading from {cwq_dev_path} ...")
        # Use the stub-loaded sparql_parser (may return None for some items)
        try:
            from utils.sparql_parser import find_reasoning_path as _frp
        except ImportError:
            _frp = lambda x: None

        with open(cwq_dev_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            path_tuples = _frp(item.get("sparql", ""))
            if not path_tuples:
                continue
            samples.append({
                "question":     item["question"],
                "topic_entity": item.get("topic_entity", ""),
                "path_strs":    [r for _, r, _, _ in path_tuples],
                "num_hops":     len(path_tuples),
            })
        print(f"[EvalDev] Loaded {len(samples)} samples from raw JSON")

    # ── Load gold answer MIDs from cwq_dev.json ───────────────────────────────
    # The cache doesn't store gold answer entity MIDs (only path rel IDs).
    # We match by question string.
    gold_mid_map = {}   # question → [gold_mids]
    if os.path.exists(cwq_dev_path):
        with open(cwq_dev_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            q    = item.get("question", "")
            mids = []
            for a in item.get("answers") or []:
                if isinstance(a, dict):
                    mid = a.get("entity_id") or a.get("id") or a.get("mid", "")
                    if mid: mids.append(mid)
                elif isinstance(a, str):
                    mids.append(a)
            if q and mids:
                gold_mid_map[q] = mids
        print(f"[EvalDev] Gold MIDs loaded for {len(gold_mid_map)} questions")

    # ── Load rel2id to reconstruct relation strings from IDs ─────────────────
    rel2id_path = os.path.join(root, "data/processed_entity/relation2id.pt")
    rel2id = torch.load(rel2id_path, map_location="cpu")
    id2rel = {v: k for k, v in rel2id.items()}

    # ── Build final items list ────────────────────────────────────────────────
    items    = []
    skipped  = 0
    no_gold  = 0

    for s in samples:
        q         = s.get("question", "")
        topic_mid = s.get("topic_entity", "")

        # Get gold MIDs
        gold_mids = gold_mid_map.get(q, [])
        if not gold_mids:
            no_gold += 1
            continue

        # Get relation strings
        if "path" in s:
            # From cache: path is list of rel_id integers
            rel_ids  = s["path"][:s["num_hops"]]
            gold_rels = [id2rel[i] for i in rel_ids if i in id2rel]
        elif "path_strs" in s:
            # From raw JSON fallback
            gold_rels = s["path_strs"]
        else:
            skipped += 1
            continue

        if not gold_rels or not topic_mid:
            skipped += 1
            continue

        items.append({
            "question":  q,
            "topic_mid": topic_mid,
            "gold_rels": gold_rels,
            "gold_mids": gold_mids,
            "num_hops":  len(gold_rels),
        })

    print(f"[EvalDev] Final: {len(items)} questions ready  "
          f"(no_gold={no_gold}, skipped={skipped})")
    return items


# ─────────────────────────────────────────────────────────────
#  Pipeline A: exp7 + exp9 traversal
# ─────────────────────────────────────────────────────────────

def traverse_pipeline_a(item: dict, agent: RLConstraintAgent,
                          tokenizer, rel2id: dict, id2rel: dict,
                          kg: dict, device: torch.device) -> dict:
    """
    Returns:
      candidate_mids : set of MID strings (entity frontier after traversal)
      path_correct   : bool (all predicted relations matched gold)
      actions_taken  : list of int actions per hop
      pred_rels      : list of predicted relation strings per hop
    """
    q = item["question"]
    enc = tokenizer(q, return_tensors="pt",
                    padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        action_logits, _, rel_logits, domain_logits = agent(
            enc["input_ids"], enc["attention_mask"])

    gold_rels    = item["gold_rels"]
    n_hops       = item["num_hops"]
    topic_mid    = item["topic_mid"]
    entity_set   = {topic_mid}
    actions_taken = []
    pred_rels     = []
    path_correct  = True

    for hop in range(min(n_hops, 4)):
        action = torch.argmax(action_logits[0, hop]).item()
        actions_taken.append(action)

        if action == ACTION_STOP:
            break

        k          = BEAM_SIZES.get(action, 1)
        topk_ids   = torch.topk(rel_logits[0, hop], k).indices.tolist()
        topk_rels  = [id2rel[i] for i in topk_ids if i in id2rel]
        pred_rels.append(topk_rels[0] if topk_rels else "")

        # Check path accuracy
        if hop < len(gold_rels) and gold_rels[hop] not in topk_rels:
            path_correct = False

        # LMDB traversal
        next_set = set()
        for mid in entity_set:
            for rel_str, target in kg.get("forward", {}).get(mid, []):
                if rel_str in topk_rels:
                    next_set.add(target)
        if next_set:
            entity_set = next_set

    return {
        "candidate_mids": entity_set,
        "path_correct":   path_correct,
        "actions_taken":  actions_taken,
        "pred_rels":      pred_rels,
    }


# ─────────────────────────────────────────────────────────────
#  Pipeline B: exp15 STRL traversal
# ─────────────────────────────────────────────────────────────

def traverse_pipeline_b(item: dict, agent: STRLAgent,
                          rel_emb_bank: RelationEmbeddingBank,
                          tokenizer, rel2id: dict, id2rel: dict,
                          kg: dict, device: torch.device) -> dict:
    q = item["question"]
    enc = tokenizer(q, return_tensors="pt",
                    padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        out = agent(enc["input_ids"], enc["attention_mask"])
        # STRLAgent.forward() returns a dict with keys:
        #   action_logits, state_values, hop_reprs, rel_logits, stop_logits, h_q
        action_logits     = out["action_logits"]
        hop_reprs_teacher = out["hop_reprs"]     # [B, 4, 1024] — already normalised

    gold_rels   = item["gold_rels"]
    n_hops      = item["num_hops"]
    topic_mid   = item["topic_mid"]
    entity_set  = {topic_mid}
    actions_taken = []
    pred_rels     = []
    path_correct  = True

    for hop in range(min(n_hops, 4)):
        action = torch.argmax(action_logits[0, hop]).item()
        actions_taken.append(action)

        if action == ACTION_STOP:
            break

        hop_repr = hop_reprs_teacher[0, hop]   # [1024] — already F.normalized in exp15

        ranked_rels = semantic_beam_with_kg_filter(
            hop_repr, rel_emb_bank, entity_set, kg, rel2id, action)

        selected_rels = [r for r, _ in ranked_rels]
        pred_rels.append(selected_rels[0] if selected_rels else "")

        if hop < len(gold_rels) and gold_rels[hop] not in selected_rels:
            path_correct = False

        # LMDB traversal
        next_set = set()
        for mid in entity_set:
            for rel_str, target in kg.get("forward", {}).get(mid, []):
                if rel_str in selected_rels:
                    next_set.add(target)
        if next_set:
            entity_set = next_set

    return {
        "candidate_mids": entity_set,
        "path_correct":   path_correct,
        "actions_taken":  actions_taken,
        "pred_rels":      pred_rels,
    }


# ─────────────────────────────────────────────────────────────
#  CDS scoring
# ─────────────────────────────────────────────────────────────

class CDSPipeline:
    """
    Wraps all three CDS stages into a single .rank() call.
    Returns the predicted answer MID (or None if candidate set is empty).
    """
    def __init__(self, s1_model, s1_tok,
                 s2_model, s2_tok,
                 s3_model, s3_tok,
                 device: torch.device,
                 mid_to_name: dict = None):
        self.s1_model   = s1_model
        self.s1_tok     = s1_tok
        self.s2_model   = s2_model
        self.s2_tok     = s2_tok
        self.s3_model   = s3_model
        self.s3_tok     = s3_tok
        self.device     = device
        self.mid_to_name = mid_to_name or {}   # MID → display name for tokenizer

    def _get_name(self, mid: str) -> str:
        return self.mid_to_name.get(mid, mid)  # fallback to MID if no name

    @torch.no_grad()
    def rank(self, question: str, candidate_mids: set,
             item_path: str = "",
             mode: str = "baseline",
             kg: dict = None,
             rel2id: dict = None,
             topic_mid: str = "",
             rel_seq: list = None,
             top_n_subgraph: int = 5,
             top_k_verify: int = 5) -> str:
        """
        mode: "baseline"  — name only
              "enriched"  — name + KG subgraph  (exp22)
              "verified"  — name only + LMDB verify after Stage 3  (exp23)
        """
        if not candidate_mids:
            return None

        cands = list(candidate_mids)
        if not cands:
            return None

        # ── Stage 1: Bi-encoder pruning → top-100 ─────────────────────────
        names  = [self._get_name(m) for m in cands]
        all_qs = [question] * len(cands)

        s1_q = self.s1_tok(all_qs, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(self.device)
        s1_e = self.s1_tok(names,  padding=True, truncation=True,
                            max_length=64,  return_tensors="pt").to(self.device)
        q_emb = self.s1_model(**s1_q).last_hidden_state[:, 0, :]
        e_emb = self.s1_model(**s1_e).last_hidden_state[:, 0, :]
        s1_scores = F.cosine_similarity(q_emb, e_emb)
        top100_idx = torch.topk(s1_scores, min(100, len(cands))).indices.tolist()
        cands_100  = [cands[i] for i in top100_idx]
        names_100  = [names[i]  for i in top100_idx]

        # ── Stage 2: Path-aware ranker → top-15 ───────────────────────────
        all_qs2 = [question]  * len(cands_100)
        paths2  = [item_path] * len(cands_100)

        s2_q = self.s2_tok(all_qs2, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(self.device)
        s2_p = self.s2_tok(paths2,  padding=True, truncation=True,
                            max_length=64,  return_tensors="pt").to(self.device)
        s2_e = self.s2_tok(names_100, padding=True, truncation=True,
                            max_length=64,  return_tensors="pt").to(self.device)
        s2_scores = self.s2_model(
            s2_q["input_ids"], s2_q["attention_mask"],
            s2_p["input_ids"], s2_p["attention_mask"],
            s2_e["input_ids"], s2_e["attention_mask"])
        top15_idx = torch.topk(s2_scores, min(15, len(cands_100))).indices.tolist()
        cands_15  = [cands_100[i] for i in top15_idx]
        names_15  = [names_100[i]  for i in top15_idx]

        # ── Stage 3: Cross-encoder → top-1 ───────────────────────────────
        if mode == "enriched" and kg and rel2id:
            doc_strs = [build_subgraph_str(m, n, kg, rel2id, top_n_subgraph)
                        for m, n in zip(cands_15, names_15)]
        else:
            doc_strs = names_15

        all_qs3 = [question] * len(cands_15)
        s3_enc  = self.s3_tok(all_qs3, doc_strs, padding=True,
                               truncation=True, max_length=256,
                               return_tensors="pt").to(self.device)
        s3_logits = self.s3_model(**s3_enc).logits.squeeze(-1)
        ranked    = torch.argsort(s3_logits, descending=True).tolist()

        # ── Optional LMDB verification (exp23 mode) ────────────────────────
        if mode == "verified" and kg and rel_seq and topic_mid:
            for rank_pos in range(min(top_k_verify, len(ranked))):
                cand_mid = cands_15[ranked[rank_pos]]
                if path_exists(topic_mid, cand_mid, rel_seq, kg):
                    return cand_mid
            # No reachable candidate found — return top-1
            return cands_15[ranked[0]]

        return cands_15[ranked[0]]


# ─────────────────────────────────────────────────────────────
#  Full evaluation loop
# ─────────────────────────────────────────────────────────────

def run_eval(args, device):
    # ── Load vocab ────────────────────────────────────────────────────────────
    rel2id  = torch.load(os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    dom2id  = torch.load(os.path.join(ROOT, "data/processed_entity/domain2id.pt"))
    id2rel  = {v: k for k, v in rel2id.items()}
    num_rel = len(rel2id); num_dom = len(dom2id)
    print(f"[EvalDev] Vocab: {num_rel} relations, {num_dom} domains")

    # ── Load KG ───────────────────────────────────────────────────────────────
    # Pipeline B (STRL) always needs the KG for semantic_beam_with_kg_filter.
    # Pipeline A only needs it for enriched/verified CDS modes.
    need_kg = (args.pipeline == "b") or (args.cds in ("enriched", "verified"))
    kg = None
    if need_kg:
        kg_path = args.kg_path or os.path.join(ROOT, "data/processed_kg/augmented_kg.pt")
        print(f"[EvalDev] Loading KG from {kg_path} ...")
        print(f"[EvalDev] (18M entity KG — takes ~2 min, loaded once per run)")
        kg = torch.load(kg_path, map_location="cpu")
        print(f"[EvalDev] KG loaded: {len(kg.get('forward',{}))} forward entities")
    else:
        print(f"[EvalDev] KG not needed for Pipeline A + {args.cds} — skipping")

    # ── Load MID→name map (optional, improves CDS readability) ───────────────
    mid_to_name = {}
    name_path = os.path.join(ROOT, "data/processed_entity/entity_names.json")
    if os.path.exists(name_path):
        with open(name_path) as f:
            for entry in json.load(f):
                mid  = entry.get("mid") or entry.get("id", "")
                name = entry.get("name", "")
                if mid and name:
                    mid_to_name[mid] = name
        print(f"[EvalDev] Loaded {len(mid_to_name)} MID→name mappings")
    else:
        print("[EvalDev] entity_names.json not found — using MIDs as display names")

    # ── Load traversal models ─────────────────────────────────────────────────
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    if args.pipeline == "a":
        print("[EvalDev] Loading Pipeline A (exp7 + exp9)...")
        base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
        exp7_ckpt  = args.exp7_ckpt or os.path.join(ROOT, "checkpoints/exp7_roberta_best.pt")
        base_model.load_state_dict(torch.load(exp7_ckpt, map_location=device))
        print(f"  exp7: {exp7_ckpt}")

        agent = RLConstraintAgent(base_model).to(device)
        exp9_ckpt = args.exp9_ckpt or os.path.join(ROOT, "checkpoints/exp9_rlmc_best.pt")
        if not os.path.exists(exp9_ckpt):
            # try to find latest epoch
            ckpts = [f for f in os.listdir(os.path.join(ROOT, "checkpoints"))
                     if f.startswith("exp9_rlmc_epoch_")]
            if ckpts:
                exp9_ckpt = os.path.join(ROOT, "checkpoints",
                                          max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0])))
        agent.load_state_dict(torch.load(exp9_ckpt, map_location=device))
        print(f"  exp9: {exp9_ckpt}")
        agent.eval()
        rel_emb_bank = None

    else:   # pipeline b
        print("[EvalDev] Loading Pipeline B (exp15 STRL)...")
        base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
        exp15_ckpt = args.exp15_ckpt or os.path.join(ROOT, "checkpoints/exp15_strl_best.pt")
        # STRLAgent wraps the planner — load planner weights first, then STRL weights
        agent = STRLAgent(base_model).to(device)
        agent.load_state_dict(torch.load(exp15_ckpt, map_location=device))
        print(f"  exp15: {exp15_ckpt}")
        agent.eval()

        rel_emb_bank = RelationEmbeddingBank(id2rel, device)
        rel_emb_bank.eval()

    # ── Load CDS models ───────────────────────────────────────────────────────
    cds = None
    if not args.recall_only:
        print("[EvalDev] Loading CDS models...")

        s1_name = "sentence-transformers/all-MiniLM-L6-v2"
        s1_tok  = AutoTokenizer.from_pretrained(s1_name)
        s1_model = AutoModel.from_pretrained(s1_name).to(device)
        s1_ckpt  = args.s1_ckpt or os.path.join(ROOT, "checkpoints/exp16v2_s1_bi.pt")
        if os.path.exists(s1_ckpt):
            s1_model.load_state_dict(torch.load(s1_ckpt, map_location=device))
            print(f"  S1: {s1_ckpt}")
        s1_model.eval()

        s2_name  = "sentence-transformers/all-mpnet-base-v2"
        s2_tok   = AutoTokenizer.from_pretrained(s2_name)
        s2_model = PathAwareRanker(model_name=s2_name).to(device)
        s2_ckpt  = args.s2_ckpt or os.path.join(ROOT, "checkpoints/exp16v2_s2_path.pt")
        if os.path.exists(s2_ckpt):
            s2_model.load_state_dict(torch.load(s2_ckpt, map_location=device))
            print(f"  S2: {s2_ckpt}")
        s2_model.eval()

        s3_name  = "BAAI/bge-reranker-base"
        s3_tok   = AutoTokenizer.from_pretrained(s3_name)
        s3_model = AutoModelForSequenceClassification.from_pretrained(s3_name).to(device)
        s3_ckpt  = args.s3_ckpt or os.path.join(ROOT, "checkpoints/exp16v2_s3_cross.pt")
        if os.path.exists(s3_ckpt):
            s3_model.load_state_dict(torch.load(s3_ckpt, map_location=device))
            print(f"  S3: {s3_ckpt}")
        s3_model.eval()

        cds = CDSPipeline(s1_model, s1_tok, s2_model, s2_tok,
                          s3_model, s3_tok, device, mid_to_name)

    # ── Load dev questions ────────────────────────────────────────────────────
    cwq_dev_path = args.cwq_dev or os.path.join(ROOT, "data/cwq_dev.json")
    dev_items    = load_cwq_dev(cwq_dev_path, ROOT)
    if args.max_questions:
        dev_items = dev_items[:args.max_questions]
        print(f"[EvalDev] Limiting to {args.max_questions} questions")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    path_correct_n   = 0
    recall_n         = 0
    hit1_n           = 0
    total_n          = 0

    action_dist = defaultdict(int)   # track action distribution

    print(f"\n[EvalDev] Evaluating {len(dev_items)} questions ...")
    print(f"  Pipeline: {'A (exp7+exp9)' if args.pipeline == 'a' else 'B (exp15 STRL)'}")
    print(f"  CDS mode: {args.cds if not args.recall_only else 'none (recall_only)'}\n")

    for item in tqdm(dev_items, desc="Evaluating"):
        total_n += 1
        gold_mids = set(item["gold_mids"])
        rel_seq   = item["gold_rels"]   # used for verification mode

        # ── Traversal ─────────────────────────────────────────────────────────
        with torch.no_grad():
            if args.pipeline == "a":
                result = traverse_pipeline_a(
                    item, agent, tokenizer, rel2id, id2rel, kg, device)
            else:
                result = traverse_pipeline_b(
                    item, agent, rel_emb_bank, tokenizer, rel2id, id2rel, kg, device)

        if result["path_correct"]:
            path_correct_n += 1

        for a in result["actions_taken"]:
            action_dist[a] += 1

        candidate_mids = result["candidate_mids"]

        # ── Reasoning Recall ──────────────────────────────────────────────────
        if candidate_mids & gold_mids:   # intersection non-empty
            recall_n += 1

        # ── CDS reranking ─────────────────────────────────────────────────────
        if cds and not args.recall_only:
            item_path = " ".join(result["pred_rels"])
            pred_mid  = cds.rank(
                question        = item["question"],
                candidate_mids  = candidate_mids,
                item_path       = item_path,
                mode            = args.cds,
                kg              = kg          if args.cds in ("enriched", "verified") else None,
                rel2id          = rel2id      if args.cds in ("enriched", "verified") else None,
                topic_mid       = item["topic_mid"],
                rel_seq         = rel_seq,
                top_n_subgraph  = args.top_n_subgraph,
                top_k_verify    = args.top_k_verify,
            )
            if pred_mid and pred_mid in gold_mids:
                hit1_n += 1

    # ── Results ───────────────────────────────────────────────────────────────
    path_acc = path_correct_n / total_n if total_n else 0.0
    recall   = recall_n       / total_n if total_n else 0.0
    hit1     = hit1_n         / total_n if total_n else 0.0

    print(f"\n{'='*55}")
    print(f"  RESULTS — Pipeline {args.pipeline.upper()}  |  CDS: {args.cds}")
    print(f"{'='*55}")
    print(f"  Questions evaluated : {total_n}")
    print(f"  Path Accuracy       : {path_acc*100:.2f}%  ({path_correct_n}/{total_n})")
    print(f"  Reasoning Recall    : {recall*100:.2f}%  ({recall_n}/{total_n})")
    if not args.recall_only:
        print(f"  Hit@1               : {hit1*100:.2f}%  ({hit1_n}/{total_n})")
    print(f"{'='*55}")

    total_actions = sum(action_dist.values())
    if total_actions > 0:
        print(f"\n  Action distribution:")
        labels = {ACTION_TIGHT: "TIGHT", ACTION_MEDIUM: "MEDIUM",
                  ACTION_LOOSE: "LOOSE", ACTION_STOP: "STOP"}
        for a, label in labels.items():
            n = action_dist[a]
            print(f"    {label:8s}: {n:5d}  ({100*n/total_actions:.1f}%)")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_dir = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    cds_tag = "recall_only" if args.recall_only else args.cds
    out_path = os.path.join(
        metrics_dir, f"eval_dev_{args.pipeline}_{cds_tag}_{ts}.csv")
    with open(out_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"pipeline,{args.pipeline}\n")
        f.write(f"cds_mode,{cds_tag}\n")
        f.write(f"total_questions,{total_n}\n")
        f.write(f"path_accuracy,{path_acc:.4f}\n")
        f.write(f"reasoning_recall,{recall:.4f}\n")
        f.write(f"hit1,{hit1:.4f}\n")
        for a, label in {ACTION_TIGHT: "TIGHT", ACTION_MEDIUM: "MEDIUM",
                          ACTION_LOOSE: "LOOSE", ACTION_STOP: "STOP"}.items():
            f.write(f"action_{label},{action_dist[a]}\n")
    print(f"\n  Results saved → {out_path}")
    return {"path_acc": path_acc, "recall": recall, "hit1": hit1}


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end dev set evaluation")
    parser.add_argument("--pipeline",       choices=["a", "b"], default="b")
    parser.add_argument("--cds",            choices=["baseline","enriched","verified"],
                        default="baseline")
    parser.add_argument("--recall_only",    action="store_true")
    parser.add_argument("--all",            action="store_true",
                        help="Run all pipeline+CDS combinations")
    parser.add_argument("--cwq_dev",        default=None)
    parser.add_argument("--kg_path",        default=None)
    parser.add_argument("--exp7_ckpt",      default=None)
    parser.add_argument("--exp9_ckpt",      default=None)
    parser.add_argument("--exp15_ckpt",     default=None)
    parser.add_argument("--s1_ckpt",        default=None)
    parser.add_argument("--s2_ckpt",        default=None)
    parser.add_argument("--s3_ckpt",        default=None)
    parser.add_argument("--max_questions",  type=int, default=None)
    parser.add_argument("--top_n_subgraph", type=int, default=5)
    parser.add_argument("--top_k_verify",   type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EvalDev] Device: {device}")

    if args.all:
        # Run all 6 meaningful combinations
        combos = [
            ("a", "baseline"), ("a", "enriched"), ("a", "verified"),
            ("b", "baseline"), ("b", "enriched"), ("b", "verified"),
        ]
        all_results = {}
        for pipeline, cds in combos:
            print(f"\n{'#'*60}")
            print(f"  Running: Pipeline {pipeline.upper()} + CDS {cds}")
            print(f"{'#'*60}")
            args.pipeline = pipeline
            args.cds      = cds
            r = run_eval(args, device)
            all_results[f"{pipeline}_{cds}"] = r

        # Summary table
        print(f"\n{'='*65}")
        print(f"  FULL RESULTS SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Combo':20s}  {'PathAcc':>9}  {'Recall':>9}  {'Hit@1':>9}")
        print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*9}")
        for combo, r in all_results.items():
            print(f"  {combo:20s}  {r['path_acc']*100:8.2f}%  "
                  f"{r['recall']*100:8.2f}%  {r['hit1']*100:8.2f}%")
        print(f"{'='*65}")

        # Save combined summary
        metrics_dir = os.path.join(ROOT, "metrics")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(metrics_dir, f"eval_dev_summary_{ts}.csv"), "w") as f:
            f.write("combo,path_acc,recall,hit1\n")
            for combo, r in all_results.items():
                f.write(f"{combo},{r['path_acc']:.4f},{r['recall']:.4f},{r['hit1']:.4f}\n")
    else:
        run_eval(args, device)


if __name__ == "__main__":
    main()
