"""
build_dpo_dataset.py — CDS Pipeline: DPO Preference Dataset Construction (Exp-37)
==================================================================================

Paper Section: §V-E  "Cascading Dual-Stage Filtering (CDS) — F3 DPO Alignment"

Purpose
-------
Converts the existing SFT training dataset (exp30_t5_mc_train.json) into the
(prompt, chosen, rejected) triplet format required by the TRL DPOTrainer used
in Exp-38 (train_f3_dpo.py).

Direct Preference Optimization (DPO) requires explicit preference pairs:
  • chosen   : the response the model SHOULD prefer  → gold entity name
  • rejected : the response the model SHOULD avoid   → a hard-negative entity name

Because the SFT dataset was already filtered through the MPNet F2 ranker (which
retains only the top-50 hardest candidates), any distractor extracted from the
prompt is already a "hard negative" — a semantically confusable entity that the
F2 ranker considered plausible.  This makes the DPO training signal particularly
sharp: the model must learn fine-grained preference among near-miss candidates.

DPO loss context (implemented in train_f3_dpo.py):
---------------------------------------------------
  L_DPO = -E[ log σ(
      β · log(π_θ(e⁺|p) / π_ref(e⁺|p))
    - β · log(π_θ(e⁻|p) / π_ref(e⁻|p))
  )]
  where:
    e⁺       = chosen  (gold entity name)
    e⁻       = rejected (hard negative entity name)
    p        = prompt  (question + candidate list)
    π_θ      = current DPO-trained policy (Flan-T5 after SFT)
    π_ref    = frozen reference policy (SFT checkpoint, Exp-26)
    β = 0.1  = KL-penalty coefficient controlling how far π_θ drifts from π_ref
    σ        = sigmoid function

  Intuitively, DPO maximises the log-ratio of preferred over rejected responses
  relative to the reference model, preventing catastrophic forgetting of the
  SFT knowledge while pushing the model to sharpen its preference.

Pipeline position
-----------------
  [train_f3_sft.py]  SFT checkpoint: exp26_t5_generative_s3.pt
        │
        ▼
  [build_dpo_dataset.py]  ← THIS FILE
        │  produces: data/exp37_t5_dpo_train.json
        ▼
  [train_f3_dpo.py]  DPO fine-tuning with beta=0.1

Inputs
------
- data/exp30_t5_mc_train.json : SFT dataset with MC prompts and gold targets
  Each record: {"prompt": str, "target": str (gold entity name)}

Outputs
-------
- data/exp37_t5_dpo_train.json : DPO preference dataset
  Each record: {"prompt": str, "chosen": str, "rejected": str}

Key design decisions
--------------------
1. Distractor extraction (extract_candidates): candidates are parsed directly
   from the prompt text rather than re-loading the underlying JSON, making this
   script self-contained and independent of the raw candidate files.
2. Random rejection sampling: one distractor is chosen uniformly at random per
   sample.  Because all distractors are hard negatives (pre-filtered by MPNet),
   the choice of *which* distractor to reject does not require a secondary
   quality signal — any of them is a valid hard negative for DPO training.
3. The DPO triplet format aligns exactly with the TRL DPOTrainer API expectation:
   keys "prompt", "chosen", "rejected".
"""

import os, sys, json, random
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# Candidate extraction from prompt text
# ──────────────────────────────────────────────────────────────────────────────

def extract_candidates(prompt):
    """
    Parse the numbered candidate list out of a listwise MC prompt string.

    The build_prompt function in train_f3_sft.py generates prompts of the form:
        1. Entity Name (Path: relation → relation)
        2. Another Entity
        ...

    This function reverses that formatting to recover the bare entity names.

    Parsing logic:
      • Lines are split and stripped.
      • A line is considered a candidate entry if its first character is a digit
        and it contains '. ' (the separator between index and name).
      • The prefix before '. ' must be purely numeric (guards against false
        positives from lines like "3D model").
      • If the name portion contains " (Path:" then everything before that marker
        is the entity name (the path annotation is stripped).

    Parameters
    ----------
    prompt : str  — the full MC prompt string

    Returns
    -------
    list of str  — candidate entity names in prompt order
    """
    names = []
    for line in prompt.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Check if line starts with a number and a dot, e.g. "1. "
        if line[0].isdigit() and '. ' in line:
            # Check if the prefix before '. ' is purely digits
            # (avoids matching lines like "3D Model. Some description")
            prefix = line.split('. ')[0]
            if prefix.isdigit():
                name_part = line.split('. ', 1)[1]
                # Strip the "(Path: ...)" annotation if present
                if ' (Path:' in name_part:
                    name = name_part.split(' (Path:')[0]
                else:
                    name = name_part
                names.append(name.strip())
    return names

# ──────────────────────────────────────────────────────────────────────────────
# Main dataset construction routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    in_path  = os.path.join(ROOT, "data/exp30_t5_mc_train.json")
    out_path = os.path.join(ROOT, "data/exp37_t5_dpo_train.json")

    print(f"Loading SFT data from {in_path}...")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dpo_data = []

    for item in tqdm(data, desc="Building DPO Pairs"):
        prompt = item["prompt"]
        # "chosen" is the gold entity name produced by the SFT training target
        chosen = item["target"]

        # Extract all candidate names from the prompt text
        cands = extract_candidates(prompt)

        # Filter out the chosen answer to get pure distractors (hard negatives).
        # These distractors were pre-filtered by the MPNet F2 ranker and are
        # therefore already semantically confusable — high-quality hard negatives.
        distractors = [c for c in cands if c != chosen]

        if not distractors:
            # If there are no distractors (unlikely), skip
            # (can happen if the prompt only listed the gold entity)
            continue

        # Randomly sample a distractor to be the rejected response.
        # Since the candidates were pre-filtered by MPNet to be the hardest 50,
        # any distractor here is a high-quality "hard negative".
        # Uniform sampling over distractors avoids introducing selection bias
        # (e.g., always picking the first distractor which might be at a
        # favourable position in the prompt).
        rejected = random.choice(distractors)

        # Construct the DPO triplet in the format expected by TRL DPOTrainer
        dpo_data.append({
            "prompt"  : prompt,    # shared context (question + candidate list)
            "chosen"  : chosen,    # preferred response: gold entity name
            "rejected": rejected   # dispreferred response: hard-negative entity
        })

    # ── Persist the DPO preference dataset ───────────────────────────────────
    print(f"\nGenerated {len(dpo_data)} DPO triplets.")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, indent=2)
    print(f"Saved to {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
