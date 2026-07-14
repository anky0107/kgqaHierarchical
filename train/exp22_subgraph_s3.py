"""
Exp 22: Subgraph Context Enrichment at Stage 3 Inference
=========================================================

NO TRAINING REQUIRED. Drop-in replacement for Stage 3 inference only.

WHY THIS MATTERS
----------------
Current Stage 3 gives BGE cross-encoder:
    question | entity_name
e.g. "Who directed Saving Private Ryan? | Steven Spielberg"

Two candidates with similar names reached via different graph paths look
IDENTICAL to BGE. It cannot distinguish them because it has no graph context.

This experiment pulls 3-5 of each candidate entity's most prominent
relations from LMDB at inference time and appends them as a short
natural-language description:

    "Steven Spielberg | directed: Saving Private Ryan, Schindler's List
     | born in: Cincinnati | type: person"

BGE can now genuinely discriminate between candidates because it sees
actual graph neighbourhood — not just a name.

WHAT CHANGES
------------
  - Stage 3 input construction only. One extra LMDB lookup per candidate.
  - Model weights: exp16v2_s3_cross.pt loaded as-is (no retraining).
  - max_length: 128 → 256 to fit the richer string.
  - Everything else (Stage 1, Stage 2, scoring) unchanged.

EXPECTED GAIN: +5–8% Hit@1  (biggest single gain available without retraining)

HOW TO RUN
----------
  python train/exp22_subgraph_s3.py

  Optional flags:
    --s3_ckpt   path to Stage 3 checkpoint  (default: exp16v2_s3_cross.pt)
    --cds_json  path to CDS eval JSON        (default: data/exp16_cds_dev.json)
    --kg_path   path to augmented_kg.pt      (default: data/processed_kg/augmented_kg.pt)
    --top_n     number of neighbour rels to surface per entity (default: 5)
    --max_length tokenizer max length        (default: 256)

OUTPUT
------
  metrics/exp22_subgraph_s3.csv   — Hit@1 baseline vs enriched, side by side
"""

import os, sys, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)


# ─────────────────────────────────────────────────────────────
#  Relation ID → natural language  (same heuristic as exp15)
# ─────────────────────────────────────────────────────────────

def _rel_to_nl(rel_id: str) -> str:
    """
    Convert Freebase relation ID to short natural language phrase.
    Uses the same heuristic as RelationEmbeddingBank._rel_to_text in
    exp15_strl.py so there is one consistent method across the codebase.

    "film.film.directed_by"     -> "film directed by"
    "people.person.nationality" -> "person nationality"
    "award.award_winner.awards_won" -> "award winner HAS awards won"
    """
    parts = rel_id.split(".")
    if len(parts) >= 3:
        subject   = parts[-2].replace("_", " ")
        predicate = parts[-1].replace("_", " ")
        if (predicate.endswith("s")
                or "owned"   in predicate
                or "founded" in predicate
                or "won"     in predicate):
            return f"{subject} HAS {predicate}"
        return f"{subject} {predicate}"
    if len(parts) == 2:
        return parts[-1].replace("_", " ")
    return rel_id.replace(".", " ").replace("_", " ")


# ─────────────────────────────────────────────────────────────
#  Subgraph description builder
# ─────────────────────────────────────────────────────────────

def build_subgraph_description(entity_mid: str, entity_name: str,
                                kg: dict, rel2id: dict,
                                top_n: int = 5) -> str:
    """
    Pull top_n most prominent relations from the entity's KG neighbourhood
    and format them as a short natural-language description.

    Strategy:
      1. Collect all (relation, target_name) pairs from forward + backward edges.
      2. Prefer forward edges (entity IS the subject) — more semantically direct.
      3. De-duplicate relation types; take top_n by frequency (most prominent).
      4. Format: "entity_name | rel_nl: target1, target2 | rel_nl2: target3"

    Falls back to entity_name only if entity_mid not in KG.

    Parameters
    ----------
    entity_mid  : Freebase MID string, e.g. "/m/01234"
    entity_name : Display name string, e.g. "Steven Spielberg"
    kg          : augmented_kg.pt dict with 'forward' and 'backward' keys
    rel2id      : relation string → integer ID mapping
    top_n       : max number of distinct relation types to surface
    """
    if not entity_mid or not kg:
        return entity_name

    forward  = kg.get("forward",  {})
    backward = kg.get("backward", {})

    # Collect (rel_str, target_name) pairs — forward edges first
    pairs = []
    for rel_str, target_mid in forward.get(entity_mid, []):
        if rel_str in rel2id:
            pairs.append((rel_str, target_mid, "fwd"))

    # Include backward edges if forward is sparse
    if len(pairs) < top_n:
        for rel_str, src_mid in backward.get(entity_mid, []):
            if rel_str in rel2id:
                pairs.append((rel_str, src_mid, "bwd"))

    if not pairs:
        return entity_name

    # Group by relation type, collect target names
    from collections import defaultdict
    rel_targets = defaultdict(list)
    for rel_str, target, direction in pairs:
        # Use the target MID as display — ideally your KG stores names,
        # otherwise use the MID directly (still useful for disambiguation)
        rel_targets[rel_str].append(target)

    # Sort by number of targets (more targets = more prominent relation)
    sorted_rels = sorted(rel_targets.items(),
                         key=lambda x: len(x[1]), reverse=True)[:top_n]

    # Build description string
    parts = [entity_name]
    for rel_str, targets in sorted_rels:
        nl      = _rel_to_nl(rel_str)
        # Show up to 3 target names per relation to keep string short
        shown   = ", ".join(str(t) for t in targets[:3])
        parts.append(f"{nl}: {shown}")

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class SubgraphCDSDataset(torch.utils.data.Dataset):
    """
    Reads the standard CDS eval JSON (same format as exp16v2).
    Expects each candidate to optionally have a 'mid' field containing
    the Freebase MID for KG lookup. Falls back to name-only if absent.

    JSON schema per item:
    {
        "question": "...",
        "candidates": [
            {
                "name":    "Steven Spielberg",
                "is_gold": true,
                "mid":     "/m/06pj8"      <- optional but needed for enrichment
            },
            ...
        ]
    }
    """
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.samples = [s for s in raw
                        if any(c["is_gold"] for c in s["candidates"])]
        print(f"[Exp22 Dataset] {len(self.samples)} samples  "
              f"from {os.path.basename(json_path)}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_passthrough(batch):
    return batch


# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, dataset: SubgraphCDSDataset,
             kg: dict, rel2id: dict,
             device: torch.device,
             mode: str = "baseline",
             top_n: int = 5,
             max_length: int = 256) -> float:
    """
    mode = "baseline"  : entity name only (original exp16v2 behaviour)
    mode = "subgraph"  : entity name + KG neighbourhood description (new)
    """
    model.eval()
    hits = 0; total = 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_passthrough)

    for batch in tqdm(loader, desc=f"Eval [{mode}]", leave=False):
        item     = batch[0]
        q        = str(item["question"])
        cands    = item["candidates"]
        if not cands:
            continue

        gold_idx = next((i for i, c in enumerate(cands) if c["is_gold"]), None)
        if gold_idx is None:
            continue

        all_qs = [q] * len(cands)

        if mode == "subgraph":
            all_es = [
                build_subgraph_description(
                    entity_mid  = c.get("mid", ""),
                    entity_name = str(c.get("name", "")),
                    kg          = kg,
                    rel2id      = rel2id,
                    top_n       = top_n,
                )
                for c in cands
            ]
        else:
            all_es = [str(c.get("name", "")) for c in cands]

        enc = tok(
            all_qs, all_es,
            padding=True, truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        logits = model(**enc).logits.squeeze(-1)
        if torch.argmax(logits).item() == gold_idx:
            hits += 1
        total += 1

    hit1 = hits / total if total > 0 else 0.0
    print(f"  [{mode}] Hit@1 = {hit1:.4f}  ({hits}/{total})")
    return hit1


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp22: Subgraph context enrichment")
    parser.add_argument("--s3_ckpt",    default=None,
                        help="Stage 3 checkpoint (default: checkpoints/exp16v2_s3_cross.pt)")
    parser.add_argument("--cds_json",   default=None,
                        help="CDS eval JSON (default: data/exp16_cds_dev.json)")
    parser.add_argument("--kg_path",    default=None,
                        help="KG path (default: data/processed_kg/augmented_kg.pt)")
    parser.add_argument("--top_n",      type=int, default=5,
                        help="Neighbour relations to surface per entity (default: 5)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Tokenizer max length (default: 256)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp22] Device: {device}")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    s3_ckpt  = args.s3_ckpt  or os.path.join(ROOT, "checkpoints/exp16v2_s3_cross.pt")
    cds_json = args.cds_json or os.path.join(ROOT, "data/exp16_cds_dev.json")
    kg_path  = args.kg_path  or os.path.join(ROOT, "data/processed_kg/augmented_kg.pt")

    for path, label in [(cds_json, "CDS JSON"), (kg_path, "KG")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # ── Load vocab ────────────────────────────────────────────────────────────
    rel2id = torch.load(
        os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    print(f"[Exp22] Loaded rel2id: {len(rel2id)} relations")

    # ── Load KG ───────────────────────────────────────────────────────────────
    print(f"[Exp22] Loading KG from {kg_path} ...")
    kg = torch.load(kg_path, map_location="cpu")
    print(f"[Exp22] KG loaded: "
          f"{len(kg.get('forward', {}))} forward entities, "
          f"{len(kg.get('backward', {}))} backward entities")

    # ── Load Stage 3 model ────────────────────────────────────────────────────
    model_name = "BAAI/bge-reranker-base"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    if os.path.exists(s3_ckpt):
        model.load_state_dict(torch.load(s3_ckpt, map_location=device))
        print(f"[Exp22] Loaded Stage 3 checkpoint: {s3_ckpt}")
    else:
        print(f"[Exp22] WARNING: checkpoint not found at {s3_ckpt}, "
              "using pretrained BGE weights only.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SubgraphCDSDataset(cds_json)

    # ── Run both modes ────────────────────────────────────────────────────────
    print(f"\n[Exp22] Running baseline (name only) ...")
    hit1_base = evaluate(model, tok, dataset, kg, rel2id, device,
                         mode="baseline", top_n=args.top_n,
                         max_length=args.max_length)

    print(f"\n[Exp22] Running subgraph enrichment (top_n={args.top_n}) ...")
    hit1_sub  = evaluate(model, tok, dataset, kg, rel2id, device,
                         mode="subgraph", top_n=args.top_n,
                         max_length=args.max_length)

    # ── Save results ──────────────────────────────────────────────────────────
    metrics_dir  = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp22_subgraph_s3.csv")
    with open(metrics_path, "w") as f:
        f.write("mode,hit1,top_n,max_length\n")
        f.write(f"baseline,{hit1_base:.4f},{args.top_n},{args.max_length}\n")
        f.write(f"subgraph,{hit1_sub:.4f},{args.top_n},{args.max_length}\n")

    delta = hit1_sub - hit1_base
    print(f"\n[Exp22] Summary:")
    print(f"  Baseline Hit@1 : {hit1_base:.4f}")
    print(f"  Subgraph Hit@1 : {hit1_sub:.4f}")
    print(f"  Delta          : {delta:+.4f}")
    print(f"  Results written to {metrics_path}")

    if delta < 0.005:
        print("\n[Exp22] NOTE: Gain is small. Check whether your CDS JSON "
              "has 'mid' fields on candidates. Without MIDs the enrichment "
              "cannot do KG lookups and falls back to name only.")


if __name__ == "__main__":
    main()
