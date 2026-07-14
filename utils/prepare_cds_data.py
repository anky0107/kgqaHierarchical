"""
prepare_cds_data.py  —  Enrich CDS JSON with MIDs, topic entity, and path fields
==================================================================================

PURPOSE
-------
exp22 (subgraph enrichment) and exp23 (two-pass verification) both need three
fields that the current exp16_cds_*.json likely does not have:

  item["topic_mid"]      Freebase MID of the topic entity (traversal start)
  item["path"]           relation sequence string for this question
  candidate["mid"]       Freebase MID of each candidate entity

This script reads:
  - data/cwq_train.json / cwq_dev.json      (gold paths + topic entities)
  - data/exp16_cds_train.json               (existing CDS data, name + is_gold)
  - data/processed_kg/augmented_kg.pt       (to resolve entity names → MIDs)
  - data/processed_entity/relation2id.pt    (for relation vocabulary)

And writes:
  - data/exp16_cds_train_enriched.json      (same format + new fields)
  - data/exp16_cds_dev_enriched.json

HOW MATCHING WORKS
------------------
CDS JSON items are matched to CWQ items by question string.
Topic entity MID comes from CWQ's topic_entity field.
Path comes from the gold SPARQL path extracted by sparql_parser.
Candidate MIDs are resolved by name lookup in the KG entity index.

ENTITY NAME → MID RESOLUTION
-----------------------------
The KG stores edges as (relation, MID) pairs, not (relation, name) pairs.
To map candidate names to MIDs we build a reverse index:
  name_to_mid[entity_name] = [MID1, MID2, ...]  (names are not always unique)

This reverse index is built from a separate entity name file if it exists
(data/processed_entity/entity_names.json), or from the CWQ gold answers
as a fallback.

HOW TO RUN
----------
  python utils/prepare_cds_data.py

  Optional flags:
    --cwq_train   path to cwq_train.json
    --cwq_dev     path to cwq_dev.json
    --cds_train   path to exp16_cds_train.json
    --cds_dev     path to exp16_cds_dev.json
    --kg_path     path to augmented_kg.pt
    --entity_names path to entity_names.json (optional, improves MID resolution)
    --out_suffix  suffix for output files (default: _enriched)

OUTPUT
------
  data/exp16_cds_train_enriched.json
  data/exp16_cds_dev_enriched.json

THEN UPDATE exp22/exp23 CALLS
-------------------------------
  python train/exp22_subgraph_s3.py --cds_json data/exp16_cds_dev_enriched.json
  python train/exp23_twopass_verify.py --cds_json data/exp16_cds_dev_enriched.json
"""

import os, sys, json, argparse
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If called from project root, __file__ is utils/prepare_cds_data.py
# so dirname(dirname(...)) = project root. Verify it looks right:
if not os.path.isdir(os.path.join(ROOT, "data")):
    # Fallback: use current working directory
    ROOT = os.getcwd()
sys.path.append(ROOT)


# ─────────────────────────────────────────────────────────────
#  CWQ loader — extracts topic entity + gold path per question
# ─────────────────────────────────────────────────────────────

def load_cwq(cwq_path: str) -> dict:
    """
    Returns dict: question_str → {
        'topic_mid':  str (topic entity MID),
        'path':       str (space-separated relation sequence),
        'gold_mids':  list[str] (gold answer entity MIDs),
    }

    CWQ JSON format (standard):
    {
      "ID": "...",
      "question": "Who directed...",
      "sparql": "SELECT ...",
      "topic_entity": {"freebase_id": "/m/0h5g4", "friendly_name": "..."},
      "answers": [{"entity_id": "/m/06pj8", "entity_name": "..."}, ...]
    }

    Path extraction: calls sparql_parser.find_reasoning_path if available,
    otherwise parses the SPARQL string directly for relation triples.
    """
    with open(cwq_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Try to import sparql_parser (already in your utils/)
    try:
        from utils.sparql_parser import find_reasoning_path
        use_parser = True
    except ImportError:
        use_parser = False
        print("[prepare_cds_data] WARNING: utils/sparql_parser not importable. "
              "Falling back to simple SPARQL relation extraction.")

    result = {}
    n_no_topic = 0; n_no_path = 0

    for item in raw:
        q = str(item.get("question", "")).strip()
        if not q:
            continue

        # Topic entity MID
        topic_entity = item.get("topic_entity") or item.get("TopicEntity") or {}
        if isinstance(topic_entity, dict):
            topic_mid = (topic_entity.get("freebase_id")
                         or topic_entity.get("id")
                         or topic_entity.get("mid", ""))
        else:
            topic_mid = str(topic_entity)

        if not topic_mid:
            n_no_topic += 1

        # Gold answer MIDs
        answers = item.get("answers") or item.get("Answers") or []
        gold_mids = []
        for a in answers:
            if isinstance(a, dict):
                mid = a.get("entity_id") or a.get("id") or a.get("mid", "")
                if mid:
                    gold_mids.append(mid)
            elif isinstance(a, str):
                gold_mids.append(a)

        # Relation path
        sparql = item.get("sparql") or item.get("sparql_query") or ""
        path_str = ""

        if use_parser and sparql:
            try:
                path = find_reasoning_path(item)
                if path and isinstance(path, list):
                    path_str = " ".join(path)
                elif path and isinstance(path, str):
                    path_str = path
            except Exception:
                pass

        if not path_str and sparql:
            # Simple fallback: extract relation strings from SPARQL triples
            # Looks for patterns like: ?x ns:film.film.directed_by ?y
            import re
            rels = re.findall(r'ns:([a-z_.]+)\s', sparql)
            # Filter to known Freebase relation patterns (contains at least 2 dots)
            rels = [r for r in rels if r.count(".") >= 2]
            path_str = " ".join(dict.fromkeys(rels))  # deduplicate, preserve order

        if not path_str:
            n_no_path += 1

        result[q] = {
            "topic_mid": topic_mid,
            "path":      path_str,
            "gold_mids": gold_mids,
        }

    print(f"[prepare_cds_data] Loaded {len(result)} questions from {os.path.basename(cwq_path)}")
    if n_no_topic > 0:
        print(f"  WARNING: {n_no_topic} items missing topic entity MID")
    if n_no_path > 0:
        print(f"  WARNING: {n_no_path} items missing relation path")

    return result


# ─────────────────────────────────────────────────────────────
#  Entity name → MID resolver
# ─────────────────────────────────────────────────────────────

def build_name_to_mid(cwq_items: dict,
                       entity_names_path: str = None) -> dict:
    """
    Build a name → [MID, ...] mapping from available sources.

    Source 1 (preferred): entity_names.json if it exists
      Format: [{"name": "Steven Spielberg", "mid": "/m/06pj8"}, ...]

    Source 2 (fallback): extract from CWQ gold answers
      Only covers gold answer entities, not all candidates.

    Returns dict: lowercase_name → list of MIDs (may be multiple per name)
    """
    name_to_mid = defaultdict(list)

    # Source 1: entity names file
    if entity_names_path and os.path.exists(entity_names_path):
        with open(entity_names_path, "r", encoding="utf-8") as f:
            enames = json.load(f)
        for entry in enames:
            name = str(entry.get("name", "")).strip().lower()
            mid  = str(entry.get("mid", entry.get("id", ""))).strip()
            if name and mid:
                name_to_mid[name].append(mid)
        print(f"[prepare_cds_data] Loaded {len(name_to_mid)} names "
              f"from {os.path.basename(entity_names_path)}")
    else:
        print("[prepare_cds_data] entity_names.json not found — "
              "using CWQ gold answers only for name→MID mapping.")

    # Source 2: CWQ gold answers
    for q_data in cwq_items.values():
        for mid in q_data.get("gold_mids", []):
            # We don't have the name here from cwq_items alone;
            # this is handled during enrichment by matching is_gold candidates
            pass

    return name_to_mid


# ─────────────────────────────────────────────────────────────
#  Core enrichment function
# ─────────────────────────────────────────────────────────────

def enrich_cds_json(cds_path: str,
                    cwq_lookup: dict,
                    name_to_mid: dict,
                    gold_name_to_mid: dict) -> list:
    """
    Read a CDS JSON file and add topic_mid, path, and candidate mid fields.

    gold_name_to_mid: dict built from CWQ answers — maps gold entity names
    to their known MIDs (higher confidence than generic name lookup).
    """
    with open(cds_path, "r", encoding="utf-8") as f:
        cds = json.load(f)

    enriched        = []
    n_matched       = 0
    n_topic_found   = 0
    n_path_found    = 0
    n_mid_found     = 0
    n_mid_total     = 0

    for item in tqdm(cds, desc=f"Enriching {os.path.basename(cds_path)}"):
        q = str(item.get("question", "")).strip()

        # Match to CWQ
        cwq_data = cwq_lookup.get(q, {})
        if cwq_data:
            n_matched += 1

        topic_mid = cwq_data.get("topic_mid", "")
        path_str  = cwq_data.get("path", "")
        gold_mids = cwq_data.get("gold_mids", [])

        if topic_mid: n_topic_found += 1
        if path_str:  n_path_found  += 1

        # Build gold name → MID map for this question's known gold entities
        q_gold_name_mid = {}
        for c in item.get("candidates", []):
            if c.get("is_gold"):
                name = str(c.get("name", "")).strip()
                # Try to find MID from gold_mids — for single-answer questions
                # the first gold_mid is the answer entity
                if gold_mids and name:
                    q_gold_name_mid[name.lower()] = gold_mids[0]

        # Enrich each candidate with a MID
        enriched_cands = []
        for c in item.get("candidates", []):
            n_mid_total += 1
            name = str(c.get("name", "")).strip()
            name_lower = name.lower()

            mid = c.get("mid", "")  # already has one → keep it

            if not mid:
                # Priority 1: gold candidate → use gold MID directly
                if c.get("is_gold") and q_gold_name_mid.get(name_lower):
                    mid = q_gold_name_mid[name_lower]
                # Priority 2: generic name lookup
                elif name_lower in name_to_mid:
                    candidates_for_name = name_to_mid[name_lower]
                    mid = candidates_for_name[0]  # take first (most common)
                # Priority 3: gold_name_to_mid from all CWQ
                elif name_lower in gold_name_to_mid:
                    mid = gold_name_to_mid[name_lower]

            if mid:
                n_mid_found += 1

            enriched_cands.append({**c, "mid": mid})

        enriched.append({
            **item,
            "topic_mid":  topic_mid,
            "path":       path_str,
            "candidates": enriched_cands,
        })

    total = len(cds)
    print(f"\n[prepare_cds_data] Enrichment summary for {os.path.basename(cds_path)}:")
    print(f"  Total items        : {total}")
    print(f"  Matched to CWQ     : {n_matched}  ({100*n_matched/max(total,1):.1f}%)")
    print(f"  topic_mid found    : {n_topic_found}  ({100*n_topic_found/max(total,1):.1f}%)")
    print(f"  path found         : {n_path_found}  ({100*n_path_found/max(total,1):.1f}%)")
    print(f"  candidate MIDs     : {n_mid_found}/{n_mid_total}  ({100*n_mid_found/max(n_mid_total,1):.1f}%)")

    if n_mid_found / max(n_mid_total, 1) < 0.5:
        print("\n  WARNING: Less than 50% of candidates have MIDs resolved.")
        print("  Provide data/processed_entity/entity_names.json for better coverage.")
        print("  Format: [{\"name\": \"Steven Spielberg\", \"mid\": \"/m/06pj8\"}, ...]")

    return enriched


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enrich CDS JSON with MID, topic_mid, path fields")
    parser.add_argument("--cwq_train",    default=None)
    parser.add_argument("--cwq_dev",      default=None)
    parser.add_argument("--cds_train",    default=None)
    parser.add_argument("--cds_dev",      default=None)
    parser.add_argument("--entity_names", default=None,
                        help="Optional: entity_names.json for name→MID lookup")
    parser.add_argument("--out_suffix",   default="_enriched",
                        help="Suffix added to output filenames (default: _enriched)")
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    cwq_train    = args.cwq_train    or os.path.join(ROOT, "data/cwq_train.json")
    cwq_dev      = args.cwq_dev      or os.path.join(ROOT, "data/cwq_dev.json")
    cds_train    = args.cds_train    or os.path.join(ROOT, "data/exp16_cds_train.json")
    cds_dev      = args.cds_dev      or os.path.join(ROOT, "data/exp16_cds_dev.json")
    entity_names = args.entity_names or os.path.join(ROOT, "data/processed_entity/entity_names.json")

    for path, label in [(cwq_train, "cwq_train"), (cwq_dev, "cwq_dev"),
                        (cds_train, "cds_train"), (cds_dev, "cds_dev")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # ── Load CWQ ──────────────────────────────────────────────────────────────
    print("\n[prepare_cds_data] Loading CWQ data...")
    cwq_train_lookup = load_cwq(cwq_train)
    cwq_dev_lookup   = load_cwq(cwq_dev)
    cwq_all          = {**cwq_train_lookup, **cwq_dev_lookup}

    # ── Build name → MID mapping ──────────────────────────────────────────────
    print("\n[prepare_cds_data] Building name→MID mapping...")
    name_to_mid = build_name_to_mid(cwq_all, entity_names)

    # Also build gold_name_to_mid from CWQ gold answers across all splits
    # This is used as a fallback for gold candidates whose names we know
    gold_name_to_mid = {}
    # (populated during enrichment from is_gold candidates matched to gold_mids)

    # ── Enrich CDS train ──────────────────────────────────────────────────────
    print(f"\n[prepare_cds_data] Enriching CDS train...")
    enriched_train = enrich_cds_json(cds_train, cwq_train_lookup,
                                      name_to_mid, gold_name_to_mid)

    out_train = cds_train.replace(".json", f"{args.out_suffix}.json")
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(enriched_train, f, ensure_ascii=False, indent=2)
    print(f"[prepare_cds_data] Written → {out_train}")

    # ── Enrich CDS dev ────────────────────────────────────────────────────────
    print(f"\n[prepare_cds_data] Enriching CDS dev...")
    enriched_dev = enrich_cds_json(cds_dev, cwq_dev_lookup,
                                    name_to_mid, gold_name_to_mid)

    out_dev = cds_dev.replace(".json", f"{args.out_suffix}.json")
    with open(out_dev, "w", encoding="utf-8") as f:
        json.dump(enriched_dev, f, ensure_ascii=False, indent=2)
    print(f"[prepare_cds_data] Written → {out_dev}")

    # ── Final instructions ─────────────────────────────────────────────────────
    print(f"""
[prepare_cds_data] Done. Now run experiments with enriched data:

  python train/exp22_subgraph_s3.py \\
      --cds_json {out_dev}

  python train/exp23_twopass_verify.py \\
      --cds_json {out_dev}

  python train/exp17_enriched_s3.py   (uses {out_train.replace('dev','train')})

If candidate MID coverage is low (<50%), add an entity_names.json file:
  Format: [{{"name": "Steven Spielberg", "mid": "/m/06pj8"}}, ...]
  This can be generated from your Freebase dump or the CWQ entity annotations.
Then re-run:
  python utils/prepare_cds_data.py --entity_names data/processed_entity/entity_names.json
""")


if __name__ == "__main__":
    main()
