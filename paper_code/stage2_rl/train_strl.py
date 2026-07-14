"""
train_strl.py — Stage 2: Semantic-Teacher RL for KGQA (STRL-KGQA) — Experiment 15
====================================================================================

PURPOSE
-------
This file implements the most sophisticated training stage in the pipeline:
**Semantic-Teacher Reinforcement Learning (STRL)**.  Instead of using gold
relation IDs as the reward signal (as Exp 9 does), STRL replaces that brittle
signal with *cosine similarity between the model's hop representation and a
frozen RoBERTa teacher's relation embedding*.  This allows the reward to capture
semantic proximity even when the exact gold relation ID is missed.

Three major components work together:
  1. **STRLAgent** — the unfrozen Exp 7 backbone + 4-action policy/value heads
     + a projection layer that maps 512-dim hop representations to the 1024-dim
     teacher embedding space.
  2. **RelationEmbeddingBank** — a frozen RoBERTa-Large that pre-encodes ALL
     Freebase relations as 1024-dim vectors (cached to disk).  This is the
     "semantic teacher" — it never changes during RL training.
  3. **calculate_semantic_rewards** — a composite reward with four components:
       R_semantic       : cosine sim between hop repr and chosen beam
       R_efficiency     : incentive for tighter beam widths
       R_connectivity   : penalises low-confidence hops likely to dead-end
       R_entity_grounded: KG-anchored reward at hop 0 — checks if chosen beam
                          intersects with relations ACTUALLY on the topic entity.
                          This catches opaque questions where semantics alone fails.

Additionally, **InfoNCE contrastive loss** (semantic_contrastive_loss) is used as
an auxiliary supervised signal alongside PPO to teach the backbone to align hop
representations with gold relation embeddings in teacher space.

PAPER SECTION
-------------
Corresponds to **Experiment 15** ("Semantic-Teacher RL") in the paper.

Key differences from Exp 9 (train_rlmc.py):
  - Exp 9  : backbone FROZEN, reward = gold-path ID hard matching
  - Exp 15 : backbone UNFROZEN (joint), reward = semantic cosine similarity
             -> no gold-path leakage, generalises to unseen entity types

PIPELINE POSITION
-----------------
  [Exp 7 checkpoint: exp7_roberta_best.pt]   <- pre-trained Stage 1 planner
           |
           v
  STRLAgent (unfrozen backbone + RL + alignment heads)
  RelationEmbeddingBank (frozen RoBERTa-Large, all relations)
  augmented_kg.pt  (KG for entity-grounded reward)
           |
           v
  Joint InfoNCE + PPO training  ->  exp15_strl_best.pt

INPUTS
------
  checkpoints/exp7_roberta_best.pt                  : Stage 1 backbone
  data/processed_entity/relation2id.pt              : dict[rel_str -> int]
  data/processed_entity/domain2id.pt                : dict[domain_str -> int]
  data/processed_entity/rel_emb_cache.pt            : cached relation embeddings
  data/processed_entity/dataset_cache_{split}.pt    : cached parsed dataset
  data/processed_kg/augmented_kg.pt                 : KG adjacency dict
  data/cwq_train.json / data/cwq_dev.json           : raw CWQ splits

OUTPUTS
-------
  checkpoints/exp15_strl_epoch_{e}.pt : checkpoint every epoch
  checkpoints/exp15_strl_best.pt      : best checkpoint by Semantic Hit@1
  metrics/exp15_strl.csv              : per-epoch metrics

KEY HYPERPARAMETERS
-------------------
  hidden_dim        = 512        (Exp 7 backbone)
  teacher_dim       = 1024       (RoBERTa-Large output / relation emb space)
  num_actions       = 4          (TIGHT / MEDIUM / LOOSE / STOP)
  BEAM_SIZES        = {0:1, 1:5, 2:50}
  REWARD_ALPHA      = 0.5        (semantic component weight)
  REWARD_BETA       = 0.3        (connectivity component weight)
  REWARD_GAMMA      = 0.2        (efficiency component weight)
  backbone_lr       = 1e-5       (slow fine-tuning of RoBERTa)
  head_lr           = 1e-4       (faster RL head adaptation)
  gamma_discount    = 0.99       (RL return discount factor)
  clip_eps          = 0.2        (PPO clipping ratio)
  entropy_coef      = 0.01       (exploration bonus)
  value_coef        = 0.5        (critic loss weight)
  temperature       = 0.07       (InfoNCE softmax temperature)
  n_negatives       = 63         (InfoNCE negatives per anchor)
  lambda_sem        = 1.0 -> 0.3 (anneals at curriculum_end=5)
  epochs            = 20
  curriculum_end    = 5          (epochs before backbone unfreezing)

HOW IT WORKS (summary)
----------------------
  Curriculum Phase (epochs 0-4):
    - Backbone FROZEN for speed.
    - lambda_sem=1.0: InfoNCE dominates, grounding semantic alignment first.
    - Weak gold bonus in reward curriculum.

  RL Refinement Phase (epochs 5-19):
    - Backbone UNFROZEN, joint fine-tuning at differential LRs.
    - lambda_sem=0.3: PPO dominates, agent learns strategic beam widths.
    - Pure semantic reward (no gold-path leakage).

  Per-batch:
    1. Rollout forward (no_grad): sample actions, compute semantic rewards.
    2. Compute discounted returns + normalised advantages.
    3. PPO forward (with grad): compute InfoNCE + clipped PPO losses jointly.
    4. Gradient clip + AdamW step.

  Dev metric — Semantic Hit@1:
    At each hop, rank all relations by cosine_sim(hop_repr, rel_emb_bank).
    Check if gold relation is rank-1. Averaged over all valid hops.
    (NOT argmax of CE logits — cosine ranking in teacher space is the true metric.)

See: notes/loss_functions_study.md for loss function rationale
See: implementation plan (conv e26e85e4) for full design doc
"""

import os, sys, json, math, functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

# ── Resolve project root for shared imports ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp7_roberta import ScaledUnifiedPlanner
from utils.sparql_parser import find_reasoning_path

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
# Discrete beam-width actions the policy chooses from at each hop
ACTION_TIGHT  = 0   # Top-1 relation  -> high precision, high risk
ACTION_MEDIUM = 1   # Top-5 relations -> balanced
ACTION_LOOSE  = 2   # Domain-wide     -> high recall, noisy
ACTION_STOP   = 3   # Terminate path

# Semantic teacher reward coefficient schedule
# (alpha, beta, gamma) = (semantic, connectivity, efficiency)
REWARD_ALPHA  = 0.5
REWARD_BETA   = 0.3
REWARD_GAMMA  = 0.2

# Maps action index to beam size k (number of top relations considered)
BEAM_SIZES = {ACTION_TIGHT: 1, ACTION_MEDIUM: 5, ACTION_LOOSE: 50}

# ─────────────────────────────────────────────────────────────────────────────
#  KG-Anchored Dataset  (replaces bare UnifiedDataset)
# ─────────────────────────────────────────────────────────────────────────────

class STRLDataset(torch.utils.data.Dataset):
    """
    Extends the base CWQ dataset with two extra fields per sample:
      topic_entity   : MID string of the starting entity
      avail_rel_ids  : set of relation IDs that actually exist on
                       the topic entity's KG neighbourhood (hop-0).

    Why: For opaque questions (e.g. 'Country Nation World Tour'),
    semantics alone can't find 'music.concert_tour.artist'.
    But the topic entity's KG edges contain it directly.
    We use this to add R_entity_grounded to the RL reward and
    to filter the semantic beam at inference time.

    Caching: parsed samples are serialised to disk so that subsequent
    runs skip the expensive SPARQL parsing and KG edge lookup.
    """
    def __init__(self, data_path, relation2id, domain2id, kg=None, max_hops=4, split="train"):
        # ── Try to load from disk cache first ─────────────────────────────────
        cache_path = os.path.join(ROOT, f"data/processed_entity/dataset_cache_{split}.pt")
        if os.path.exists(cache_path):
            print(f"[Dataset] Loading cached {split} samples from {cache_path}...")
            self.samples = torch.load(cache_path)
            return

        # ── Build from scratch if no cache ────────────────────────────────────
        print(f"[Dataset] Processing {split} data from {data_path}...")
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = []
        from tqdm import tqdm
        from utils.sparql_parser import find_reasoning_path
        for item in tqdm(data, desc=f"Processing {split}"):
            # ── Parse SPARQL into hop-by-hop relation path ─────────────────────
            path = find_reasoning_path(item.get('sparql', ''))
            if path is None:
                continue   # skip questions with unparseable SPARQL

            # ── Derive Freebase domain from first relation prefix ──────────────
            main_rel = path[0][1]
            domain   = main_rel.split('.')[0] if '.' in main_rel else 'none'
            if domain not in domain2id:
                domain = 'none'

            # ── Convert relation strings to integer IDs, skip OOV ─────────────
            rel_ids, valid = [], True
            for _, rel, _, _ in path:
                if rel in relation2id:
                    rel_ids.append(relation2id[rel])
                else:
                    valid = False; break
            if not valid:
                continue

            # ── Pad or truncate path to max_hops ──────────────────────────────
            num_hops = len(rel_ids)
            if num_hops > max_hops:
                rel_ids = rel_ids[:max_hops]
            else:
                rel_ids = rel_ids + [0] * (max_hops - num_hops)

            # ── KG-anchored: get relations available on topic entity ───────────
            # Look up which relations actually exist on the topic entity in the KG.
            # These are used for R_entity_grounded reward at hop 0.
            topic_mid   = item.get('topic_entity', None) or (path[0][0] if path else None)
            avail_set   = set()
            if topic_mid and kg is not None:
                for rel, _ in kg.get('forward',  {}).get(topic_mid, []):
                    if rel in relation2id:
                        avail_set.add(relation2id[rel])
                for rel, _ in kg.get('backward', {}).get(topic_mid, []):
                    if rel in relation2id:
                        avail_set.add(relation2id[rel])

            self.samples.append({
                'question':     item['question'],
                'domain':       domain2id[domain],
                'path':         rel_ids,
                'num_hops':     min(num_hops, max_hops),
                'topic_entity': topic_mid or '',
                'avail_rels':   list(avail_set),   # list of int rel IDs on topic entity
            })

        # ── Save cache to disk ────────────────────────────────────────────────
        print(f"[Dataset] Done. Saving cache to {cache_path}...")
        torch.save(self.samples, cache_path)

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_strl(batch, tokenizer):
    """
    Collate function for STRL dataset.  Tokenises questions and stacks tensors.
    avail_rels is kept as a list-of-lists (CPU) since it's variable length.

    Returns:
        encoded    : BERT/RoBERTa BatchEncoding  [B, seq]
        domains    : [B]            gold domain IDs
        paths      : [B, max_hops] gold relation IDs
        nums       : [B]            number of valid hops
        avail_rels : list[list[int]]  per-sample KG edge rel IDs (variable length)
    """
    questions  = [s['question'] for s in batch]
    domains    = torch.tensor([s['domain']   for s in batch])
    paths      = torch.tensor([s['path']     for s in batch])
    nums       = torch.tensor([s['num_hops'] for s in batch])
    encoded    = tokenizer(questions, padding=True, truncation=True,
                           max_length=128, return_tensors='pt')
    # avail_rels: variable length per sample — keep as list-of-lists
    avail_rels = [s['avail_rels'] for s in batch]
    return encoded, domains, paths, nums, avail_rels


# ─────────────────────────────────────────────────────────────────────────────
#  Component 1: Relation Embedding Bank (Frozen Teacher)
# ─────────────────────────────────────────────────────────────────────────────

class RelationEmbeddingBank(nn.Module):
    """
    Pre-computes and caches RoBERTa-Large embeddings for every Freebase relation.

    This is the "frozen semantic teacher": its [N_rel, 1024] matrix defines
    what each relation *means* semantically.  The RL agent's hop representations
    are rewarded when they align (cosine similarity) with these embeddings.

    The bank is computed ONCE at training start, cached to disk, and then
    held as a non-trainable buffer (`register_buffer`) for fast GPU dot products.
    It is never back-propagated through — it is always `.eval()` and frozen.
    """
    def __init__(self, id2rel: dict, device, batch_size: int = 64):
        super().__init__()
        self.device = device
        self.id2rel = id2rel
        N = len(id2rel)

        # ── Try to load from disk cache ────────────────────────────────────────
        cache_path = os.path.join(ROOT, "data/processed_entity/rel_emb_cache.pt")
        if os.path.exists(cache_path):
            print(f"[RelEmb] Loading cached embeddings from {cache_path}...")
            emb_matrix = torch.load(cache_path, map_location=device)
            if emb_matrix.shape[0] == N:
                # Shape matches — use cached matrix directly as a non-trainable buffer
                self.register_buffer("emb_matrix", emb_matrix)
                return
            print("[RelEmb] Cache mismatch, re-computing...")

        # ── Build embeddings with a fresh RoBERTa-Large (then discard it) ─────
        print(f"[RelEmb] Pre-computing {N} relation embeddings with frozen RoBERTa...")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        encoder   = RobertaModel.from_pretrained("roberta-large").to(device)
        encoder.eval()   # frozen, no dropout

        # Convert integer IDs back to human-readable relation text
        rel_texts = [self._rel_to_text(id2rel[i]) for i in range(N)]
        embs = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                batch_texts = rel_texts[start : start + batch_size]
                enc = tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=32, return_tensors="pt"
                ).to(device)
                out = encoder(**enc)
                # Use CLS token as the relation embedding [B, 1024]
                cls = out.last_hidden_state[:, 0, :]   # [B, 1024]
                embs.append(cls.cpu())

        emb_matrix = torch.cat(embs, dim=0).to(device)  # [N, 1024]
        # Register as non-trainable buffer so it moves with the module (e.g. .cuda())
        self.register_buffer("emb_matrix", emb_matrix)

        # Save cache for future runs
        torch.save(emb_matrix.cpu(), cache_path)
        print(f"[RelEmb] Done. Matrix shape: {self.emb_matrix.shape} (Saved to cache)")

        # Free encoder GPU memory — it's no longer needed
        del encoder

    @staticmethod
    def _rel_to_text(rel_id: str) -> str:
        """
        Improved directional mapping:
        'music.concert_tour.artist' -> 'concert tour artist'
        'music.artist.concert_tours' -> 'artist concert tours'
        We also strip the top-level 'common' or 'base' prefixes to reduce noise.

        The heuristic detects inverse relations by checking if the predicate part
        ends in 's' or contains past-tense verbs ('owned', 'founded'), converting
        those to 'HAS <predicate>' phrasing for more natural embeddings.
        """
        parts = rel_id.split(".")
        if len(parts) > 1:
            # Check if it looks like an inverse relation by part frequency
            # This is a heuristic: relations ending in 's' or 'ed' often represent 'has' or 'was'
            subject   = parts[-2].replace("_", " ")
            predicate = parts[-1].replace("_", " ")
            if predicate.endswith("s") or "owned" in predicate or "founded" in predicate:
                return f"{subject} HAS {predicate}"
            else:
                return f"{subject} {predicate}"
        return rel_id.replace(".", " ").replace("_", " ")

    def get(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve embeddings for a subset of relations.
        rel_ids: [K] — indices of relations in the beam
        returns: [K, 1024] embeddings on the correct device
        """
        return self.emb_matrix[rel_ids].to(self.device)

    def all(self) -> torch.Tensor:
        """Returns full [N, 1024] matrix on device for full-relation cosine ranking."""
        return self.emb_matrix.to(self.device)


# ─────────────────────────────────────────────────────────────────────────────
#  Component 2: STRL Agent (unfrozen backbone + RL heads)
# ─────────────────────────────────────────────────────────────────────────────

class STRLAgent(nn.Module):
    """
    Semantic-Teacher RL Agent.

    Builds on top of ScaledUnifiedPlanner (Exp7) but:
    - Backbone is UNFROZEN (joint training with lr=1e-5)
    - Adds a 4-action policy head + value head
    - forward() also returns hop_repr for teacher scoring
    - Projection layer aligns hop_repr (hidden=512) to teacher space (1024)

    Architecture overview:
      input_ids, attention_mask
           |
      RoBERTa-Large (encoder)     ← UNFROZEN from epoch 5 onward
           |
      Linear(1024, 512)  (proj)   ← projects CLS to hidden_dim
           |
      + hop_embeddings  [max_hops, 512]
           |
      TransformerEncoder(2 layers, nhead=8)
           |
      hop_repr  [B, max_hops, 512]
         / | \\
        /  |  \\
    policy value  proj_to_teacher
    [B,H,4] [B,H] [B,H,1024] <- normalised, for reward + InfoNCE
    """
    def __init__(self, base_model: ScaledUnifiedPlanner):
        super().__init__()
        self.base = base_model
        hidden_dim = 512   # Exp7 hidden dim

        # UNFROZEN backbone — critical difference from Exp9
        # All base parameters receive gradients (selectively frozen in train loop
        # during curriculum phase for speed, then re-enabled at epoch 5)
        for param in self.base.parameters():
            param.requires_grad = True

        # ── Policy head: 4 actions per hop ────────────────────────────────────
        # GELU + Dropout for regularisation (vs plain ReLU in Exp 9)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

        # ── Value head (for PPO advantage) ────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # ── Alignment projection: hop_repr -> teacher space ───────────────────
        # Maps 512-dim hop repr to 1024-dim relation embedding space so that
        # InfoNCE loss and cosine rewards can operate in the same space as
        # the frozen teacher's relation embeddings.
        self.proj_to_teacher = nn.Linear(hidden_dim, 1024)

    def forward(self, input_ids, attention_mask):
        """
        Single forward pass that computes everything needed for both PPO and InfoNCE.
        Called TWICE per training batch: once with no_grad (rollout) and once
        with grad (PPO + InfoNCE update).

        Returns (dict):
          action_logits : [B, max_hops, 4]     raw policy scores per action
          state_values  : [B, max_hops]         value estimates per hop
          hop_reprs     : [B, max_hops, 1024]   L2-normalised in teacher space
          rel_logits    : [B, max_hops, num_rel] base model relation scores
          stop_logits   : [B, max_hops]          base model stop scores
          h_q           : [B, 512]               question embedding
        """

        # We need refined_repr [B, H, 512] and h_q [B, 512]
        # Since ScaledUnifiedPlanner forward doesn't return them by default,
        # we can either modify it or re-calculate ONLY the projector/transformer parts.
        # To avoid modifying the base class file, we re-calculate the small parts here
        # but RE-USE the encoder output from the base forward pass if possible.

        # HOWEVER, the base.forward() doesn't expose the encoder output.
        # OPTIMIZATION: We modify the base class forward slightly to return these or
        # we just accept one re-calculation of the encoder but ENSURE we only do it ONCE.

        # Actually, let's just re-calculate the whole thing ONCE here and not call self.base()
        # This is the cleanest way to avoid redundant RoBERTa calls.

        # ── RoBERTa encoding: CLS token as question repr ──────────────────────
        enc_out = self.base.encoder(input_ids, attention_mask)
        q_h     = enc_out.last_hidden_state[:, 0, :]   # [B, 1024] RoBERTa CLS
        h_q     = self.base.proj(q_h)                  # [B, 512] projected

        # ── Unified Planner logic (reproduced from base to expose internals) ───
        # Add hop positional embeddings then run cross-hop Transformer
        init     = h_q.unsqueeze(1) + self.base.hop_embeddings    # [B, H, 512]
        hop_repr = self.base.transformer(init)                     # [B, H, 512]

        # Base model output heads (needed for compatibility with existing eval code)
        rel_logits  = self.base.relation_head(hop_repr)
        stop_logits = self.base.adaptive_stop_head(hop_repr).squeeze(-1)

        # ── RL heads: policy + value ──────────────────────────────────────────
        action_logits = self.policy_head(hop_repr)               # [B, H, 4]
        state_values  = self.value_head(hop_repr).squeeze(-1)    # [B, H]

        # ── Teacher-space projection for semantic reward + InfoNCE ────────────
        # Project to 1024-dim and L2-normalise so cosine sim = dot product
        hop_repr_teacher = self.proj_to_teacher(hop_repr)        # [B, H, 1024]
        hop_repr_teacher = F.normalize(hop_repr_teacher, dim=-1) # unit vectors

        return {
            "action_logits": action_logits,
            "state_values":  state_values,
            "hop_reprs":     hop_repr_teacher,   # normalised teacher-space reprs
            "rel_logits":    rel_logits,
            "stop_logits":   stop_logits,
            "h_q":           h_q,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Component 3: Reward Calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_semantic_rewards(
    actions:        torch.Tensor,         # [B, max_hops] int64
    hop_reprs:      torch.Tensor,         # [B, max_hops, 1024] normalised
    rel_emb_bank:   RelationEmbeddingBank,
    path_lengths:   torch.Tensor,         # [B]
    gold_paths:     torch.Tensor,         # [B, max_hops] rel IDs
    epoch:          int,
    curriculum_end: int = 5,
    avail_rels:     list = None,          # list[list[int]] — per-sample available rel IDs
) -> torch.Tensor:
    """
    Computes the STRL reward tensor [B, max_hops].

    Reward = alpha*R_semantic + beta*R_connectivity_proxy + gamma*R_efficiency
           + R_entity_grounded            <- NEW: KG-anchored signal
           + (weak gold bonus if epoch < curriculum_end)

    R_semantic:
      Cosine sim between hop_repr and top-k relation embeddings.
      TIGHT->1, MEDIUM->5, LOOSE->50

    R_entity_grounded (the key addition for opaque questions):
      At hop 0, check if the chosen beam intersects with relations
      that ACTUALLY EXIST on the topic entity in the KG.
      - +0.6 if beam & avail_rels != empty  (beam is reachable from entity)
      - -0.4 if beam & avail_rels = empty   (beam leads to a dead end)
      This is what catches 'Country Nation World Tour' type questions
      where pure semantics fails but KG edges give the answer.

    R_connectivity proxy: +0.3 if good sim, -0.5 if likely dead end
    R_efficiency: TIGHT +0.3 | MEDIUM 0.0 | LOOSE -0.3
    Curriculum: +0.2 weak gold bonus for epochs < curriculum_end
    """
    device = actions.device
    B, max_hops = actions.size()
    rewards = torch.zeros(B, max_hops, device=device)

    # Pre-fetch full relation embedding matrix [N_rel, 1024]
    # All dot products below are equivalent to cosine similarity (both normalised)
    all_rel_embs = rel_emb_bank.all()   # [N_rel, 1024]
    N_rel = all_rel_embs.size(0)

    use_curriculum = (epoch < curriculum_end)

    for b in range(B):
        L = int(path_lengths[b].item())   # true path length for this sample

        for h in range(max_hops):
            action = int(actions[b, h].item())

            # ── Past true path length: only STOP is correct ───────────────────
            if h >= L:
                rewards[b, h] = +0.8 if action == ACTION_STOP else -0.8
                continue

            # ── Within valid path ─────────────────────────────────────────────
            if action == ACTION_STOP:
                rewards[b, h] = -1.0  # Stopped too early
                continue

            # ── Determine beam size from action ───────────────────────────────
            k = BEAM_SIZES.get(action, 5)
            k = min(k, N_rel)   # cap to vocabulary size

            # Hop representation in teacher space [1024]
            hop_vec = hop_reprs[b, h]   # already normalised

            # ── Cosine sim with all relations via dot product ─────────────────
            # all_rel_embs: [N, 1024], hop_vec: [1024]
            # Since both are L2-normalised, dot product = cosine similarity
            sims        = torch.mv(all_rel_embs, hop_vec)    # [N_rel]
            top_indices = torch.topk(sims, k).indices        # [k] highest-sim rel IDs
            top_sims    = sims[top_indices]                   # [k] their cosine scores

            # ── R_semantic: max sim within chosen beam ────────────────────────
            # The best-matching relation in the beam determines semantic quality
            r_sem = REWARD_ALPHA * top_sims.max().item()

            # ── R_efficiency ──────────────────────────────────────────────────
            # Incentivise tighter beams (TIGHT is most efficient, LOOSE is not)
            if action == ACTION_TIGHT:
                r_eff = REWARD_GAMMA * 0.3
            elif action == ACTION_MEDIUM:
                r_eff = 0.0
            else:  # LOOSE
                r_eff = REWARD_GAMMA * (-0.3)

            # ── R_connectivity proxy ──────────────────────────────────────────
            # If max cosine sim < 0.3, the beam likely leads to a dead end in KG
            best_sim = top_sims.max().item()
            if best_sim < 0.3:
                r_conn = REWARD_BETA * (-0.5)  # likely dead end
            else:
                r_conn = REWARD_BETA * 0.3     # good connectivity signal

            # ── R_entity_grounded (KG-anchored) ──────────────────────────────
            # Only meaningful at hop 0: does the chosen beam contain
            # a relation that exists on the topic entity in the KG?
            # This catches opaque questions where semantics fails but
            # the entity's edges reveal the correct path.
            r_entity = 0.0
            if h == 0 and avail_rels is not None:
                entity_rel_set = set(avail_rels[b])     # pre-computed KG edges
                beam_rel_set   = set(top_indices.tolist())
                if entity_rel_set and beam_rel_set:
                    if entity_rel_set & beam_rel_set:   # intersection non-empty
                        r_entity = +0.6   # beam contains a reachable KG edge
                    else:
                        r_entity = -0.4   # beam leads to unreachable relations

            # ── Curriculum: weak gold bonus ───────────────────────────────────
            # During early epochs, provide a mild gold-path supervision signal
            # to help the policy bootstrap before purely semantic rewards suffice.
            r_curriculum = 0.0
            if use_curriculum:
                gold_rel_id = int(gold_paths[b, h].item())
                if gold_rel_id in top_indices.tolist():
                    r_curriculum = 0.2   # gold relation is in the chosen beam

            rewards[b, h] = r_sem + r_eff + r_conn + r_entity + r_curriculum

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
#  Component 4a: InfoNCE Semantic Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────

def semantic_contrastive_loss(
    hop_reprs:    torch.Tensor,           # [B, max_hops, 1024] normalised, WITH grad
    gold_paths:   torch.Tensor,           # [B, max_hops] gold relation IDs
    path_lengths: torch.Tensor,           # [B]
    rel_emb_bank: RelationEmbeddingBank,  # frozen [N_rel, 1024]
    temperature:  float = 0.07,
    n_negatives:  int   = 63,             # negatives per anchor (total batch = 64)
) -> torch.Tensor:
    """
    InfoNCE contrastive loss that trains proj_to_teacher to produce
    hop representations semantically aligned with the gold relation.

    For each valid (b, h) pair:
      anchor   = hop_reprs[b, h]                  — projected question repr (WITH grad)
      positive = rel_emb_bank[gold_paths[b, h]]   — frozen teacher embedding (no grad)
      negatives = n_negatives random relations sampled from the bank (no grad)

    Loss = -log [ exp(sim(anchor, pos)/tau) / sum exp(sim(anchor, neg_i)/tau) ]
         = cross_entropy(logits, label=0)  where position 0 = the positive

    Temperature tau=0.07 is very small (standard in SimCLR / CLIP style training),
    making the loss very sensitive and forcing accurate alignment.

    This directly teaches: "given the question + hop context,
    your internal representation should point at the correct relation type."
    It's the gradient that makes the RL agent semantically aware.
    """
    device   = hop_reprs.device
    all_embs = rel_emb_bank.all()    # [N_rel, 1024], frozen, no grad
    N_rel    = all_embs.size(0)

    losses = []
    for b in range(hop_reprs.size(0)):
        L = int(path_lengths[b].item())
        for h in range(L):                        # only iterate over valid hops
            anchor   = hop_reprs[b, h]            # [1024] WITH gradient
            gold_id  = int(gold_paths[b, h].item())
            positive = all_embs[gold_id]          # [1024], no grad (frozen teacher)

            # ── Sample n_negatives random relation indices ─────────────────────
            neg_ids = torch.randint(0, N_rel, (n_negatives,), device=device)
            # Replace any accidental collision with gold to avoid false negatives
            neg_ids[neg_ids == gold_id] = (gold_id + 1) % N_rel
            negatives = all_embs[neg_ids]         # [n_neg, 1024], no grad

            # ── Stack positive first, then negatives ───────────────────────────
            # Convention: label 0 = first slot = the positive pair
            candidates = torch.cat([positive.unsqueeze(0), negatives], dim=0)  # [64, 1024]

            # ── Scaled dot products (cosine sim since both normalised) ─────────
            # anchor has grad; candidates do not -> gradient flows through anchor
            logits = torch.mv(candidates, anchor) / temperature   # [64]

            # Label 0 = positive is always first in candidates
            target = torch.zeros(1, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    if not losses:
        # All samples had zero valid hops (degenerate batch) — return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Component 4b: PPO Update (now receives InfoNCE loss jointly)
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    agent: STRLAgent,
    optimizer: torch.optim.Optimizer,
    old_log_probs:     torch.Tensor,   # [B, max_hops] — from rollout (detached)
    actions:           torch.Tensor,   # [B, max_hops]
    advantages:        torch.Tensor,   # [B, max_hops] — normalised
    returns:           torch.Tensor,   # [B, max_hops] — discounted
    new_action_logits: torch.Tensor,   # [B, max_hops, 4]  — from fresh forward pass
    new_state_values:  torch.Tensor,   # [B, max_hops]     — from fresh forward pass
    l_semantic:        torch.Tensor,   # scalar — InfoNCE loss (already computed)
    lambda_sem:        float = 1.0,    # weight of semantic loss (anneals over curriculum)
    clip_eps:          float = 0.2,    # PPO clipping ratio (standard: 0.2)
    entropy_coef:      float = 0.01,   # exploration bonus coefficient
    value_coef:        float = 0.5,    # critic loss weight
) -> dict:
    """
    Joint PPO + InfoNCE update.

    L_total = lambda_sem * L_infonce
            + L_actor          (clipped surrogate objective)
            + value_coef * L_critic  (value function regression)
            - entropy_coef * entropy (exploration bonus)

    The clipped PPO surrogate prevents destructively large policy updates:
      ratio = pi_new(a|s) / pi_old(a|s)  (importance weight)
      surr1 = ratio * A(s,a)
      surr2 = clip(ratio, 1-eps, 1+eps) * A(s,a)
      L_actor = -min(surr1, surr2)    (pessimistic bound)

    The InfoNCE term is what connects PPO to semantics:
    it forces the backbone to learn WHAT the question is asking for
    (semantic alignment), while PPO learns HOW WIDE the beam should be
    (strategic efficiency).

    Gradient clipping (max_norm=1.0) prevents exploding gradients from the
    joint loss, which can be unstable when both RoBERTa and RL heads update.
    """
    # ── Recompute log-probs and entropy from the fresh forward pass ───────────
    dist          = torch.distributions.Categorical(logits=new_action_logits)
    new_log_probs = dist.log_prob(actions)   # [B, max_hops]
    entropy       = dist.entropy().mean()    # scalar — higher = more exploration

    # ── PPO clipped surrogate objective ──────────────────────────────────────
    ratio  = torch.exp(new_log_probs - old_log_probs)   # importance sampling ratio
    surr1  = ratio * advantages
    surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    # Take the pessimistic (minimum) of the two surrogates to bound policy change
    actor_loss  = -torch.min(surr1, surr2).mean()

    # ── Critic loss: regress estimated values to actual discounted returns ─────
    critic_loss = F.mse_loss(new_state_values, returns)

    # ── Combined loss ──────────────────────────────────────────────────────────
    total_loss = (
        lambda_sem * l_semantic     # semantic alignment (InfoNCE)
        + actor_loss                # policy gradient (PPO)
        + value_coef * critic_loss  # value function (A2C critic)
        - entropy_coef * entropy    # exploration bonus (negative = maximise H)
    )

    # Backprop with gradient clipping for stability
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "actor_loss":    actor_loss.item(),
        "critic_loss":   critic_loss.item(),
        "entropy":       entropy.item(),
        "semantic_loss": l_semantic.item(),
        "total_loss":    total_loss.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_exp15_strl():
    """
    Main training loop for STRL-KGQA (Exp 15).

    Two-phase curriculum:
      Phase 1 (epochs 0 to curriculum_end-1):
        - RoBERTa backbone FROZEN for 3x speedup.
        - lambda_sem=1.0: InfoNCE dominates, grounds semantic alignment first.
        - Reward includes weak gold bonus (curriculum).
      Phase 2 (epoch curriculum_end onward):
        - RoBERTa backbone UNFROZEN, differential learning rates.
        - lambda_sem=0.3: PPO dominates, agent refines strategic beam widths.
        - Pure semantic reward (no gold-path supervision leak).

    Dev metric is Semantic Hit@1 (rank gold relation by cosine sim in teacher
    space, check if it's top-1).  This is the paper's primary evaluation metric
    for this experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp15] Device: {device}")

    # ── Load Mappings ──────────────────────────────────────────────────────────
    data_dir = os.path.join(ROOT, "data/processed_entity")
    rel2id   = torch.load(os.path.join(data_dir, "relation2id.pt"))
    dom2id   = torch.load(os.path.join(data_dir, "domain2id.pt"))
    id2rel   = {v: k for k, v in rel2id.items()}   # reverse map for embedding bank
    num_rel  = len(rel2id)
    num_dom  = len(dom2id)
    print(f"[Exp15] Vocab: {num_rel} relations, {num_dom} domains")

    # ── Load Exp7 Base Model ───────────────────────────────────────────────────
    # Start from the supervised Stage 1 checkpoint — the RL fine-tuning builds on it
    print("[Exp15] Loading Exp7 backbone (exp7_roberta_best.pt)...")
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    ckpt_path  = os.path.join(ROOT, "checkpoints/exp7_roberta_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Exp7 checkpoint not found at {ckpt_path}.\n"
            "Run train/exp7_roberta.py first."
        )
    base_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("[Exp15] Exp7 weights loaded.")

    # ── Pre-compute Relation Embedding Bank (frozen teacher) ───────────────────
    # This is computed ONCE and cached; subsequent runs reload from disk instantly
    rel_emb_bank = RelationEmbeddingBank(id2rel, device).to(device)
    rel_emb_bank.eval()   # always inference mode — never updated
    for p in rel_emb_bank.parameters():
        p.requires_grad = False   # extra safety: freeze everything in the bank

    # ── Build STRL Agent ───────────────────────────────────────────────────────
    agent = STRLAgent(base_model).to(device)
    print(f"[Exp15] Agent parameters: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")

    # ── Differential Learning Rates ────────────────────────────────────────────
    # Backbone (RoBERTa): 1e-5 — slow, stable
    # Policy/Value heads: 1e-4 — faster RL adaptation
    # Alignment proj:     1e-4
    backbone_params = list(agent.base.encoder.parameters()) + list(agent.base.proj.parameters())
    head_params = (
        list(agent.policy_head.parameters())
        + list(agent.value_head.parameters())
        + list(agent.proj_to_teacher.parameters())
        + list(agent.base.hop_embeddings.__iter__() if False else [agent.base.hop_embeddings])
        + list(agent.base.transformer.parameters())
        + list(agent.base.relation_head.parameters())
        + list(agent.base.domain_head.parameters())
        + list(agent.base.confidence_head.parameters())
        + list(agent.base.adaptive_stop_head.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5, "weight_decay": 0.01},
        {"params": head_params,     "lr": 1e-4, "weight_decay": 0.01},
    ])

    # ── Load KG for entity-grounded reward ────────────────────────────────────
    # The KG provides per-entity adjacency lists used in R_entity_grounded reward
    print("[Exp15] Loading KG for entity-grounded reward...")
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    kg = torch.load(kg_path, map_location='cpu')
    print(f"[Exp15] KG loaded: {len(kg.get('forward', {}))} forward entities")

    # ── Dataset (KG-anchored) ──────────────────────────────────────────────────
    tokenizer    = RobertaTokenizer.from_pretrained("roberta-large")
    train_ds     = STRLDataset(os.path.join(ROOT, "data/cwq_train.json"), rel2id, dom2id, kg, split="train")
    dev_ds       = STRLDataset(os.path.join(ROOT, "data/cwq_dev.json"),   rel2id, dom2id, kg, split="dev")
    collate      = functools.partial(collate_strl, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=8,  shuffle=False, collate_fn=collate)
    print(f"[Exp15] Dataset: {len(train_ds)} train, {len(dev_ds)} dev samples")

    # ── Metrics Setup ──────────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "metrics"),      exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"),  exist_ok=True)
    metrics_path = os.path.join(ROOT, "metrics/exp15_strl.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,avg_reward,actor_loss,critic_loss,semantic_loss,entropy,dev_sem_hit1\n")

    # ── Training Config ────────────────────────────────────────────────────────
    epochs          = 20
    gamma           = 0.99   # discount factor for computing discounted returns
    curriculum_end  = 5      # switch to pure semantic reward after epoch 5
    clip_eps        = 0.2    # PPO clipping ratio
    scaler          = torch.amp.GradScaler("cuda")

    # lambda_sem schedule: 1.0 during grounding phase, 0.3 during RL refinement
    # This ensures semantic understanding is established BEFORE RL shapes strategy
    def get_lambda_sem(ep: int) -> float:
        return 1.0 if ep < curriculum_end else 0.3

    # ── Resume Logic ───────────────────────────────────────────────────────────
    # Automatically detect and resume from the latest epoch checkpoint
    start_epoch  = 0
    best_dev_acc = 0.0
    exp15_ckpts  = [f for f in os.listdir(os.path.join(ROOT, 'checkpoints'))
                    if f.startswith('exp15_strl_epoch_') and f.endswith('.pt')]
    if exp15_ckpts:
        latest_ckpt = max(exp15_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0]) + 1
        ckpt_path   = os.path.join(ROOT, 'checkpoints', latest_ckpt)
        print(f"[Exp15] Resuming from {ckpt_path} (Starting Epoch {start_epoch})")
        agent.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Restore best_dev_acc so that the checkpoint saving logic works correctly
        if os.path.exists(metrics_path):
            import pandas as pd
            try:
                df = pd.read_csv(metrics_path)
                if not df.empty:
                    best_dev_acc = df['dev_sem_hit1'].max()
                    print(f"[Exp15] Loaded best_dev_acc from metrics: {best_dev_acc*100:.2f}%")
            except Exception: pass

    print(f"\n[Exp15] Starting STRL Training - {epochs} epochs")
    print(f"  Grounding phase  : epochs 0-{curriculum_end-1} -> lambda_sem=1.0 + weak gold bonus")
    print(f"  RL refinement    : epochs {curriculum_end}-{epochs-1} -> lambda_sem=0.3 + pure semantic")
    print(f"  Dev metric       : Semantic Hit@1 (cosine sim rank, NOT CE logit argmax)")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        agent.train()

        # ── Speed Optimization: Partial Freezing ──────────────────────────────
        # During the grounding phase (epochs 0-5), we freeze the
        # RoBERTa backbone. This speeds up training by ~3x and
        # focuses the model on learning the policy and alignment
        # heads first.
        if epoch < curriculum_end:
            print(f"  [Speed] Epoch {epoch}: Backbone FROZEN (Grounding Phase)")
            for p in agent.base.encoder.parameters():
                p.requires_grad = False
        else:
            print(f"  [Speed] Epoch {epoch}: Backbone UNFROZEN (Refinement Phase)")
            for p in agent.base.encoder.parameters():
                p.requires_grad = True

        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        # Accumulate per-epoch statistics for logging
        ep_rewards       = []
        ep_actor_loss    = []
        ep_critic_loss   = []
        ep_semantic_loss = []
        ep_entropy       = []
        lambda_sem       = get_lambda_sem(epoch)

        for enc, doms, gold_paths, path_lengths, avail_rels in t_bar:
            enc          = enc.to(device)
            doms         = doms.to(device)
            gold_paths   = gold_paths.to(device)      # [B, max_hops]
            path_lengths = path_lengths.to(device)    # [B]
            # avail_rels stays as list-of-lists (CPU) — used in reward calc only

            # ── Rollout (no grad for sampling) ────────────────────────────────
            # First forward pass: just sample actions and compute rewards.
            # No gradient needed here — actions and rewards are treated as fixed data.
            with torch.no_grad():
                fwd = agent(enc["input_ids"], enc["attention_mask"])

            action_logits = fwd["action_logits"]   # [B, H, 4]
            state_values  = fwd["state_values"]    # [B, H]
            hop_reprs     = fwd["hop_reprs"]       # [B, H, 1024]

            # Sample discrete actions from the policy distribution
            dist          = torch.distributions.Categorical(logits=action_logits)
            actions       = dist.sample()                # [B, H]
            old_log_probs = dist.log_prob(actions)       # [B, H]

            # ── Compute Rewards ───────────────────────────────────────────────
            # Semantic rewards based on cosine sim + KG-grounded signal.
            # Done under no_grad since rewards are constants for the PPO update.
            with torch.no_grad():
                rewards = calculate_semantic_rewards(
                    actions       = actions,
                    hop_reprs     = hop_reprs,
                    rel_emb_bank  = rel_emb_bank,
                    path_lengths  = path_lengths,
                    gold_paths    = gold_paths,
                    epoch         = epoch,
                    curriculum_end= curriculum_end,
                    avail_rels    = avail_rels,   # KG-anchored: entity's actual edges
                )

            ep_rewards.append(rewards.mean().item())

            # ── Compute Returns & Advantages ──────────────────────────────────
            # Discounted return: G_t = r_t + gamma * G_{t+1}  (backward sweep)
            B, H   = rewards.size()
            returns    = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)

            for b in range(B):
                G = 0.0
                for h in reversed(range(H)):   # walk backward through hops
                    G              = rewards[b, h].item() + gamma * G
                    returns[b, h]  = G
                    # Baseline subtraction: A = G - V(s) reduces return variance
                    advantages[b, h] = G - state_values[b, h].detach().item()

            # Normalise advantages (reduces variance, stabilises PPO updates)
            adv_mean   = advantages.mean()
            adv_std    = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # ── Joint PPO + InfoNCE Update (fresh forward WITH grad) ──────────
            # This forward pass produces hop_reprs WITH gradient so that
            # InfoNCE can backprop through proj_to_teacher -> transformer -> RoBERTa
            with torch.amp.autocast("cuda"):
                fwd2 = agent(enc["input_ids"], enc["attention_mask"])

                # InfoNCE: teach the backbone to understand relation semantics
                # hop_reprs from fwd2 have gradient — this is the key difference
                l_sem = semantic_contrastive_loss(
                    hop_reprs    = fwd2["hop_reprs"],    # [B,H,1024] WITH grad
                    gold_paths   = gold_paths,
                    path_lengths = path_lengths,
                    rel_emb_bank = rel_emb_bank,         # frozen teacher
                    temperature  = 0.07,
                )

                update_stats = ppo_update(
                    agent             = agent,
                    optimizer         = optimizer,
                    old_log_probs     = old_log_probs.detach(),  # stop grad from rollout
                    actions           = actions,
                    advantages        = advantages,
                    returns           = returns,
                    new_action_logits = fwd2["action_logits"],
                    new_state_values  = fwd2["state_values"],
                    l_semantic        = l_sem,
                    lambda_sem        = lambda_sem,
                    clip_eps          = clip_eps,
                )

            ep_actor_loss.append(update_stats["actor_loss"])
            ep_critic_loss.append(update_stats["critic_loss"])
            ep_semantic_loss.append(update_stats["semantic_loss"])
            ep_entropy.append(update_stats["entropy"])

            t_bar.set_postfix(
                rew  = f"{rewards.mean().item():.3f}",
                sem  = f"{update_stats['semantic_loss']:.3f}",
                act  = f"{update_stats['actor_loss']:.3f}",
                ent  = f"{update_stats['entropy']:.3f}",
            )

        # ── Dev Evaluation: Semantic Hit@1 ────────────────────────────────────
        # We rank relations by cosine_sim(hop_repr, rel_emb_bank) — NOT logit argmax.
        # This is the true test of semantic understanding.
        agent.eval()
        sem_hit1_total, sem_hit1_correct = 0, 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        all_rel_embs  = rel_emb_bank.all()   # [N_rel, 1024]

        with torch.no_grad():
            for enc, doms, gold_paths, path_lengths, _ in dev_loader:
                enc          = enc.to(device)
                gold_paths   = gold_paths.to(device)
                path_lengths = path_lengths.to(device)

                fwd = agent(enc["input_ids"], enc["attention_mask"])
                hop_reprs_dev = fwd["hop_reprs"]          # [B, H, 1024] normalised

                # Collect action distribution stats for logging
                actions_dev = fwd["action_logits"].argmax(dim=-1)
                for a in actions_dev.flatten().tolist():
                    action_counts[a] = action_counts.get(a, 0) + 1

                B = gold_paths.size(0)
                for b in range(B):
                    L = int(path_lengths[b].item())
                    for h in range(L):
                        hop_vec  = hop_reprs_dev[b, h]              # [1024]
                        sims     = torch.mv(all_rel_embs, hop_vec)  # [N_rel] cosine sims
                        pred_rel = sims.argmax().item()             # semantic top-1
                        gold_rel = int(gold_paths[b, h].item())
                        sem_hit1_correct += int(pred_rel == gold_rel)
                        sem_hit1_total   += 1

        # ── Epoch aggregation + logging ───────────────────────────────────────
        dev_acc  = sem_hit1_correct / sem_hit1_total if sem_hit1_total > 0 else 0.0
        avg_r    = sum(ep_rewards)       / len(ep_rewards)
        avg_al   = sum(ep_actor_loss)    / len(ep_actor_loss)
        avg_cl   = sum(ep_critic_loss)   / len(ep_critic_loss)
        avg_sl   = sum(ep_semantic_loss) / len(ep_semantic_loss)
        avg_ent  = sum(ep_entropy)       / len(ep_entropy)

        print(f"\n[Ep {epoch:02d}] Reward={avg_r:.4f} | Sem={avg_sl:.4f} | "
              f"Actor={avg_al:.4f} | Critic={avg_cl:.4f} | "
              f"Entropy={avg_ent:.4f} | SemanticHit@1={dev_acc*100:.2f}%")
        print(f"  Action dist -> TIGHT:{action_counts[0]} MEDIUM:{action_counts[1]} "
              f"LOOSE:{action_counts[2]} STOP:{action_counts[3]}")
        mode = f"GROUNDING (lambda_sem={lambda_sem:.1f})" if epoch < curriculum_end else f"RL-REFINE (lambda_sem={lambda_sem:.1f})"
        print(f"  Mode: {mode}")

        # ── Metrics Log ───────────────────────────────────────────────────────
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_r:.4f},{avg_al:.4f},{avg_cl:.4f},{avg_sl:.4f},{avg_ent:.4f},{dev_acc:.4f}\n")

        # ── Checkpoint ────────────────────────────────────────────────────────
        # Save every epoch for resumability
        torch.save(agent.state_dict(),
                   os.path.join(ROOT, f"checkpoints/exp15_strl_epoch_{epoch}.pt"))

        # Additionally save the best model by Semantic Hit@1
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(agent.state_dict(),
                       os.path.join(ROOT, "checkpoints/exp15_strl_best.pt"))
            print(f"  * New best Semantic Hit@1: {dev_acc*100:.2f}% - saved exp15_strl_best.pt")

    print("\n[Exp15] Training complete.")
    print(f"  Best dev path accuracy: {best_dev_acc*100:.2f}%")
    print(f"  Best checkpoint: checkpoints/exp15_strl_best.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Inference Helper: Semantic Beam + KG Filter
# ─────────────────────────────────────────────────────────────────────────────

def semantic_beam_with_kg_filter(
    hop_repr:    torch.Tensor,          # [1024] normalised query vector
    rel_emb_bank: RelationEmbeddingBank,
    current_entity_set: set,            # MIDs reachable at this hop
    kg:          dict,                  # augmented KG {'forward':..., 'backward':...}
    rel2id:      dict,
    action:      int,                   # TIGHT/MEDIUM/LOOSE
) -> list:
    """
    Semantic beam selection with KG-grounded fallback.
    Used at inference time to generate actual candidate relations for traversal.

    Step 1 — Semantic beam:
      Rank all relations by cosine_sim(hop_repr, rel_emb_bank).
      Take top-k according to action (TIGHT=1, MEDIUM=5, LOOSE=50).

    Step 2 — KG intersection:
      Find which of those top-k relations ACTUALLY EXIST on the
      current entity set in the KG.
      If intersection is non-empty -> use it (pure semantic + reachable).

    Step 3 — KG-only fallback:
      If semantic beam has ZERO reachable relations (opaque question
      like 'Country Nation World Tour'), fall back to ALL relations
      available on the entity and re-rank those by semantic score.
      This is the fix for blind-shooting on bridge-entity questions.

    Returns: list of (relation_name, score) sorted by score desc
    """
    # ── Step 1: Semantic beam — rank all relations by cosine sim ──────────────
    all_embs = rel_emb_bank.all()                       # [N_rel, 1024]
    sims     = torch.mv(all_embs, hop_repr)             # [N_rel] dot products (= cosine)
    k        = BEAM_SIZES.get(action, 5)
    k        = min(k, all_embs.size(0))
    top_k    = torch.topk(sims, k)
    sem_ids  = top_k.indices.tolist()                   # semantic top-k rel IDs
    sem_sims = top_k.values.tolist()

    # ── Step 2: Collect relations actually reachable from current entities ─────
    reachable_rels = set()
    for mid in current_entity_set:
        for rel, _ in kg.get('forward',  {}).get(mid, []):
            if rel in rel2id:
                reachable_rels.add(rel2id[rel])
        for rel, _ in kg.get('backward', {}).get(mid, []):
            if rel in rel2id:
                reachable_rels.add(rel2id[rel])

    id2rel_local = {v: k for k, v in rel2id.items()}

    # Intersection of semantic top-k and KG-reachable relations
    intersection = [(rid, s) for rid, s in zip(sem_ids, sem_sims)
                    if rid in reachable_rels]

    if intersection:
        # Best case: semantic beam contains reachable relations — use them
        return [(id2rel_local[rid], s) for rid, s in intersection]

    # ── Step 3: Fallback — rank reachable relations by semantic score ──────────
    # Triggered for opaque questions where semantic top-k leads to dead ends.
    # Take ALL reachable relations and rank them by cosine sim to hop_repr.
    if reachable_rels:
        fallback = [(rid, sims[rid].item()) for rid in reachable_rels]
        fallback.sort(key=lambda x: x[1], reverse=True)
        return [(id2rel_local[rid], s) for rid, s in fallback[:k]]

    # Nothing reachable at all — return semantic beam as last resort
    return [(id2rel_local[rid], s) for rid, s in zip(sem_ids, sem_sims)]


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_exp15_strl()
