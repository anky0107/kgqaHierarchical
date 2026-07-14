"""
Exp 20: Hard Negative Mining (Full 27k Dataset)
================================================

Extracts the top-ranked WRONG candidates from Stage 1 + Stage 2 to use as
hard negatives for Stage 3 training (Exp 23).

Fixes vs the original exp18 script:
  - Loads the full 27k dataset instead of the 2k prototype set.
  - Caps raw candidate beams at MAX_CANDS_INPUT (5000) before S1 to prevent
    GPU OOM / hangs on pathologically large beams (some have 50k+ candidates).
    Gold candidates are always preserved regardless of the cap.
  - Checkpointing: writes a partial .jsonl file every CHECKPOINT_EVERY items
    so the run can be resumed if interrupted (--resume flag).
"""

import os, sys, json, torch, random, argparse
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from cds_pipeline.pipeline import CDSPipeline
from cds_pipeline.utils import flatten_path

# ── Tunable constants ─────────────────────────────────────────────────────────
MAX_CANDS_INPUT   = 5000   # cap raw candidates passed to S1 (keep all golds)
CHECKPOINT_EVERY  = 500    # write checkpoint every N processed items
# ─────────────────────────────────────────────────────────────────────────────


def load_checkpoint(ckpt_path):
    """Load already-processed items from a .jsonl checkpoint file."""
    if not os.path.exists(ckpt_path):
        return [], set()
    done_questions = set()
    done_items = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done_items.append(item)
                done_questions.add(item["question"])
            except json.JSONDecodeError:
                pass
    print(f"[Resume] Loaded {len(done_items)} already-processed items from checkpoint.")
    return done_items, done_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if it exists.")
    args = parser.parse_args()

    train_path = os.path.join(ROOT, "data/exp16_cds_train_full.json")
    out_path   = os.path.join(ROOT, "data/exp18_cds_train_hard_full.json")
    ckpt_path  = out_path.replace(".json", ".ckpt.jsonl")

    print(f"Loading Exp16 CDS Train dataset (full 27k) from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples.")

    # -- Resume from checkpoint if requested ----------------------------------
    hard_data = []
    skip_questions = set()
    if args.resume and os.path.exists(ckpt_path):
        hard_data, skip_questions = load_checkpoint(ckpt_path)
        print(f"  Skipping {len(skip_questions)} already-processed questions.")

    # -- Filter to unprocessed items ------------------------------------------
    pending = [item for item in data if item["question"] not in skip_questions]
    print(f"  {len(pending)} items remaining to process.")

    if not pending:
        print("All items already processed. Writing final output...")
    else:
        # Initialize pipeline with reduced top-k — mining only needs 15 hard negs
        # so S1=50 is plenty (saves ~75% of bi-encoder work vs s1_top_k=200).
        pipeline = CDSPipeline(s1_top_k=50, s2_top_k=15)

        processed_since_ckpt = 0

        for item in tqdm(pending, desc="Mining Hard Negatives"):
            q     = item["question"]
            cands = item["candidates"]

            # Skip if no gold
            golds_in_original = [c for c in cands if c.get("is_gold")]
            if not golds_in_original:
                continue

            path_str = flatten_path(item.get("path"))

            # ── Cap large beams to prevent OOM (preserve all golds) ──────────
            if len(cands) > MAX_CANDS_INPUT:
                negs = [c for c in cands if not c.get("is_gold")]
                negs = random.sample(negs, min(MAX_CANDS_INPUT - len(golds_in_original), len(negs)))
                cands_for_pipeline = golds_in_original + negs
            else:
                cands_for_pipeline = cands

            # ── Run S1 and S2 ────────────────────────────────────────────────
            with torch.no_grad():
                s1_cands = pipeline._stage1(q, cands_for_pipeline)
                s2_cands = pipeline._stage2(q, path_str, s1_cands)

            # ── Extract hard negatives ───────────────────────────────────────
            hard_negs  = [c for c in s2_cands if not c.get("is_gold")]
            final_cands = golds_in_original + hard_negs
            final_cands = final_cands[:16]

            item["candidates"] = final_cands
            hard_data.append(item)
            processed_since_ckpt += 1

            # ── Write checkpoint ─────────────────────────────────────────────
            if processed_since_ckpt >= CHECKPOINT_EVERY:
                with open(ckpt_path, "a", encoding="utf-8") as f:
                    for d in hard_data[-processed_since_ckpt:]:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                print(f"\n  [Checkpoint] Saved {len(hard_data)} total items so far.")
                processed_since_ckpt = 0

        # Flush remaining items to checkpoint
        if processed_since_ckpt > 0:
            with open(ckpt_path, "a", encoding="utf-8") as f:
                for d in hard_data[-processed_since_ckpt:]:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"\n  [Checkpoint] Saved final {len(hard_data)} total items.")

    # -- Write final JSON output -----------------------------------------------
    print(f"\nSaving {len(hard_data)} hard negative samples to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard_data, f)   # no indent to keep file size reasonable
    print("Done!")

    # -- Clean up checkpoint if successful ------------------------------------
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Checkpoint file removed: {ckpt_path}")


if __name__ == "__main__":
    main()
