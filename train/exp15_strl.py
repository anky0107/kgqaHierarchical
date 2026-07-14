"""
Exp 15: Semantic-Teacher RL for KGQA (STRL-KGQA)
=================================================

Architecture:
  Stage 1 — Exp7 backbone (RoBERTa-Large), UNFROZEN, joint training
  Stage 2 — RL Policy head (4 actions: TIGHT/MEDIUM/LOOSE/STOP)
             Reward = α·R_semantic + β·R_connectivity + γ·R_efficiency
  Stage 3 — Cross-Encoder Answer Selector (replaces random set selection)

Key difference from Exp9:
  - Exp9: backbone FROZEN, reward = gold-path ID matching
  - Exp15: backbone UNFROZEN (joint), reward = SEMANTIC TEACHER cosine similarity
           → no gold-path leakage, generalises to unseen entity types

Loss Design (what makes the RL agent understand semantics):
  L_infonce   : InfoNCE contrastive loss — trains proj_to_teacher so that
                hop_repr[b,h] points at gold relation embedding in 1024-dim space.
                This is the core supervised signal that teaches MEANING.
  L_ppo       : Clipped PPO (actor + critic) — trains policy to pick right
                beam width given the model's confidence.
  L_total     : λ_sem * L_infonce + L_ppo
                λ_sem anneals from 1.0 → 0.3 over curriculum epochs so that
                supervised semantic grounding comes first, then RL refines.

Training curriculum:
  Epochs 0-4  : λ_sem=1.0, weak gold bonus in reward (grounding phase)
  Epochs 5-10 : λ_sem=0.3, pure semantic reward (RL refinement phase)

Dev metric — Semantic Hit@1:
  At each hop, rank all relations by cosine_sim(hop_repr, rel_emb_bank).
  Check if gold relation is rank-1. Averaged over all valid hops.
  (NOT argmax of CE logits — the old blind metric.)

See: notes/loss_functions_study.md for loss function rationale
See: implementation plan (conv e26e85e4) for full design doc"""

import os, sys, json, math, functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp7_roberta import ScaledUnifiedPlanner
from utils.sparql_parser import find_reasoning_path

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
ACTION_TIGHT  = 0   # Top-1 relation  → high precision, high risk
ACTION_MEDIUM = 1   # Top-5 relations → balanced
ACTION_LOOSE  = 2   # Domain-wide     → high recall, noisy
ACTION_STOP   = 3   # Terminate path

# Semantic teacher reward coefficient schedule
# (α, β, γ) = (semantic, connectivity, efficiency)
REWARD_ALPHA  = 0.5
REWARD_BETA   = 0.3
REWARD_GAMMA  = 0.2

BEAM_SIZES = {ACTION_TIGHT: 1, ACTION_MEDIUM: 5, ACTION_LOOSE: 50}

# ─────────────────────────────────────────────────────────────
#  KG-Anchored Dataset  (replaces bare UnifiedDataset)
# ─────────────────────────────────────────────────────────────

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
    """
    def __init__(self, data_path, relation2id, domain2id, kg=None, max_hops=4, split="train"):
        cache_path = os.path.join(ROOT, f"data/processed_entity/dataset_cache_{split}.pt")
        if os.path.exists(cache_path):
            print(f"[Dataset] Loading cached {split} samples from {cache_path}...")
            self.samples = torch.load(cache_path)
            return

        print(f"[Dataset] Processing {split} data from {data_path}...")
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        from tqdm import tqdm
        from utils.sparql_parser import find_reasoning_path
        for item in tqdm(data, desc=f"Processing {split}"):
            path = find_reasoning_path(item.get('sparql', ''))
            if path is None:
                continue

            main_rel = path[0][1]
            domain   = main_rel.split('.')[0] if '.' in main_rel else 'none'
            if domain not in domain2id:
                domain = 'none'

            rel_ids, valid = [], True
            for _, rel, _, _ in path:
                if rel in relation2id:
                    rel_ids.append(relation2id[rel])
                else:
                    valid = False; break
            if not valid:
                continue

            num_hops = len(rel_ids)
            if num_hops > max_hops:
                rel_ids = rel_ids[:max_hops]
            else:
                rel_ids = rel_ids + [0] * (max_hops - num_hops)

            # ── KG-anchored: get relations available on topic entity ──
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
                'avail_rels':   list(avail_set),   # list of int rel IDs
            })
            
        print(f"[Dataset] Done. Saving cache to {cache_path}...")
        torch.save(self.samples, cache_path)

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_strl(batch, tokenizer):
    questions  = [s['question'] for s in batch]
    domains    = torch.tensor([s['domain']   for s in batch])
    paths      = torch.tensor([s['path']     for s in batch])
    nums       = torch.tensor([s['num_hops'] for s in batch])
    encoded    = tokenizer(questions, padding=True, truncation=True,
                           max_length=128, return_tensors='pt')
    # avail_rels: variable length per sample — keep as list-of-lists
    avail_rels = [s['avail_rels'] for s in batch]
    return encoded, domains, paths, nums, avail_rels

# ─────────────────────────────────────────────────────────────
#  Component 1: Relation Embedding Bank (Frozen Teacher)
# ─────────────────────────────────────────────────────────────

class RelationEmbeddingBank(nn.Module):
    def __init__(self, id2rel: dict, device, batch_size: int = 64):
        super().__init__()
        self.device = device
        self.id2rel = id2rel
        N = len(id2rel)
        
        cache_path = os.path.join(ROOT, "data/processed_entity/rel_emb_cache.pt")
        if os.path.exists(cache_path):
            print(f"[RelEmb] Loading cached embeddings from {cache_path}...")
            emb_matrix = torch.load(cache_path, map_location=device)
            if emb_matrix.shape[0] == N:
                self.register_buffer("emb_matrix", emb_matrix)
                return
            print("[RelEmb] Cache mismatch, re-computing...")

        print(f"[RelEmb] Pre-computing {N} relation embeddings with frozen RoBERTa...")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        encoder   = RobertaModel.from_pretrained("roberta-large").to(device)
        encoder.eval()

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
                cls = out.last_hidden_state[:, 0, :]   # [B, 1024]
                embs.append(cls.cpu())

        emb_matrix = torch.cat(embs, dim=0).to(device)  # [N, 1024]
        self.register_buffer("emb_matrix", emb_matrix)
        
        # Save cache
        torch.save(emb_matrix.cpu(), cache_path)
        print(f"[RelEmb] Done. Matrix shape: {self.emb_matrix.shape} (Saved to cache)")

        # Free encoder
        del encoder

    @staticmethod
    def _rel_to_text(rel_id: str) -> str:
        """
        Improved directional mapping:
        'music.concert_tour.artist' -> 'concert tour artist'
        'music.artist.concert_tours' -> 'artist concert tours'
        We also strip the top-level 'common' or 'base' prefixes to reduce noise.
        """
        parts = rel_id.split(".")
        if len(parts) > 1:
            # Check if it looks like an inverse relation by part frequency
            # This is a heuristic: relations ending in 's' or 'ed' often represent 'has' or 'was'
            subject = parts[-2].replace("_", " ")
            predicate = parts[-1].replace("_", " ")
            if predicate.endswith("s") or "owned" in predicate or "founded" in predicate:
                return f"{subject} HAS {predicate}"
            else:
                return f"{subject} {predicate}"
        return rel_id.replace(".", " ").replace("_", " ")

    def get(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        rel_ids: [K] — indices of relations in the beam
        returns: [K, 1024] embeddings on the correct device
        """
        return self.emb_matrix[rel_ids].to(self.device)

    def all(self) -> torch.Tensor:
        """Returns full [N, 1024] matrix on device."""
        return self.emb_matrix.to(self.device)


# ─────────────────────────────────────────────────────────────
#  Component 2: STRL Agent (unfrozen backbone + RL heads)
# ─────────────────────────────────────────────────────────────

class STRLAgent(nn.Module):
    """
    Semantic-Teacher RL Agent.

    Builds on top of ScaledUnifiedPlanner (Exp7) but:
    - Backbone is UNFROZEN (joint training with lr=1e-5)
    - Adds a 4-action policy head + value head
    - forward() also returns hop_repr for teacher scoring
    - Projection layer aligns hop_repr (hidden=512) to teacher space (1024)
    """
    def __init__(self, base_model: ScaledUnifiedPlanner):
        super().__init__()
        self.base = base_model
        hidden_dim = 512   # Exp7 hidden dim

        # UNFROZEN backbone — critical difference from Exp9
        for param in self.base.parameters():
            param.requires_grad = True

        # ── Policy head: 4 actions per hop ──────────────────
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

        # ── Value head (for PPO advantage) ──────────────────
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # ── Alignment projection: hop_repr → teacher space ──
        # Maps 512-dim hop repr to 1024-dim relation embedding space
        self.proj_to_teacher = nn.Linear(hidden_dim, 1024)

    def forward(self, input_ids, attention_mask):
        """
        Returns:
          action_logits : [B, max_hops, 4]
          state_values  : [B, max_hops]
          hop_reprs     : [B, max_hops, 1024]  — in teacher space
          rel_logits    : [B, max_hops, num_rel]
          h_q           : [B, 512]  — question embedding
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
        
        enc_out = self.base.encoder(input_ids, attention_mask)
        q_h     = enc_out.last_hidden_state[:, 0, :]
        h_q     = self.base.proj(q_h)                            # [B, 512]
        
        # Unified Planner logic
        init     = h_q.unsqueeze(1) + self.base.hop_embeddings    # [B, H, 512]
        hop_repr = self.base.transformer(init)                   # [B, H, 512]
        
        rel_logits  = self.base.relation_head(hop_repr)
        stop_logits = self.base.adaptive_stop_head(hop_repr).squeeze(-1)
        
        # RL heads
        action_logits = self.policy_head(hop_repr)               # [B, H, 4]
        state_values  = self.value_head(hop_repr).squeeze(-1)    # [B, H]

        # Project to teacher space for semantic reward
        hop_repr_teacher = self.proj_to_teacher(hop_repr)        # [B, H, 1024]
        hop_repr_teacher = F.normalize(hop_repr_teacher, dim=-1)

        return {
            "action_logits": action_logits,
            "state_values":  state_values,
            "hop_reprs":     hop_repr_teacher,
            "rel_logits":    rel_logits,
            "stop_logits":   stop_logits,
            "h_q":           h_q,
        }


# ─────────────────────────────────────────────────────────────
#  Component 3: Reward Calculator
# ─────────────────────────────────────────────────────────────

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

    Reward = α·R_semantic + β·R_connectivity_proxy + γ·R_efficiency
           + R_entity_grounded            ← NEW: KG-anchored signal
           + (weak gold bonus if epoch < curriculum_end)

    R_semantic:
      Cosine sim between hop_repr and top-k relation embeddings.
      TIGHT→1, MEDIUM→5, LOOSE→50

    R_entity_grounded (the key addition for opaque questions):
      At hop 0, check if the chosen beam intersects with relations
      that ACTUALLY EXIST on the topic entity in the KG.
      - +0.6 if beam ∩ avail_rels ≠ ∅  (beam is reachable from entity)
      - -0.4 if beam ∩ avail_rels = ∅  (beam leads to a dead end)
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
    all_rel_embs = rel_emb_bank.all()   # [N_rel, 1024]
    N_rel = all_rel_embs.size(0)

    use_curriculum = (epoch < curriculum_end)

    for b in range(B):
        L = int(path_lengths[b].item())

        for h in range(max_hops):
            action = int(actions[b, h].item())

            # ── Past true path length: only STOP is correct ─────────
            if h >= L:
                rewards[b, h] = +0.8 if action == ACTION_STOP else -0.8
                continue

            # ── Within valid path ────────────────────────────────────
            if action == ACTION_STOP:
                rewards[b, h] = -1.0  # Stopped too early
                continue

            # Determine beam size
            k = BEAM_SIZES.get(action, 5)
            k = min(k, N_rel)

            # Hop representation in teacher space [1024]
            hop_vec = hop_reprs[b, h]   # already normalised

            # Cosine sim with all relations — dot product since both normalised
            # all_rel_embs: [N, 1024], hop_vec: [1024]
            sims = torch.mv(all_rel_embs, hop_vec)    # [N_rel]
            top_indices = torch.topk(sims, k).indices  # [k]
            top_sims    = sims[top_indices]             # [k]

            # ── R_semantic: max sim within chosen beam ───────────────
            r_sem = REWARD_ALPHA * top_sims.max().item()

            # ── R_efficiency ─────────────────────────────────────────
            if action == ACTION_TIGHT:
                r_eff = REWARD_GAMMA * 0.3
            elif action == ACTION_MEDIUM:
                r_eff = 0.0
            else:  # LOOSE
                r_eff = REWARD_GAMMA * (-0.3)

            # ── R_connectivity proxy ─────────────────────────────────
            best_sim = top_sims.max().item()
            if best_sim < 0.3:
                r_conn = REWARD_BETA * (-0.5)
            else:
                r_conn = REWARD_BETA * 0.3

            # ── R_entity_grounded (KG-anchored) ──────────────────────
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
                        r_entity = +0.6
                    else:
                        r_entity = -0.4                 # beam unreachable from entity

            # ── Curriculum: weak gold bonus ──────────────────────────
            r_curriculum = 0.0
            if use_curriculum:
                gold_rel_id = int(gold_paths[b, h].item())
                if gold_rel_id in top_indices.tolist():
                    r_curriculum = 0.2

            rewards[b, h] = r_sem + r_eff + r_conn + r_entity + r_curriculum

    return rewards


# ─────────────────────────────────────────────────────────────
#  Component 4a: InfoNCE Semantic Contrastive Loss
# ─────────────────────────────────────────────────────────────

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
      anchor   = hop_reprs[b, h]                  — projected question repr
      positive = rel_emb_bank[gold_paths[b, h]]   — frozen teacher embedding
      negatives = n_negatives random relations sampled from the bank

    Loss = -log [ exp(sim(anchor, pos)/τ) / Σ exp(sim(anchor, neg_i)/τ) ]

    This directly teaches: "given the question + hop context,
    your internal representation should point at the correct relation type."
    It's the gradient that makes the RL agent semantically aware.
    """
    device    = hop_reprs.device
    all_embs  = rel_emb_bank.all()    # [N_rel, 1024], frozen, no grad
    N_rel     = all_embs.size(0)

    losses = []
    for b in range(hop_reprs.size(0)):
        L = int(path_lengths[b].item())
        for h in range(L):                        # only valid hops
            anchor   = hop_reprs[b, h]            # [1024]
            gold_id  = int(gold_paths[b, h].item())
            positive = all_embs[gold_id]          # [1024], no grad

            # Sample n_negatives random relation indices (excluding gold)
            neg_ids = torch.randint(0, N_rel, (n_negatives,), device=device)
            # Replace any accidental gold hit
            neg_ids[neg_ids == gold_id] = (gold_id + 1) % N_rel
            negatives = all_embs[neg_ids]         # [n_neg, 1024], no grad

            # Stack positives + negatives: first slot is positive
            candidates = torch.cat([positive.unsqueeze(0), negatives], dim=0)  # [64, 1024]

            # Dot products (both are already L2-normalised)
            logits = torch.mv(candidates, anchor) / temperature   # [64]

            # Label 0 = positive is always first
            target = torch.zeros(1, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────
#  Component 4b: PPO Update (now receives InfoNCE loss jointly)
# ─────────────────────────────────────────────────────────────

def ppo_update(
    agent: STRLAgent,
    optimizer: torch.optim.Optimizer,
    old_log_probs:     torch.Tensor,   # [B, max_hops] — from rollout
    actions:           torch.Tensor,   # [B, max_hops]
    advantages:        torch.Tensor,   # [B, max_hops]
    returns:           torch.Tensor,   # [B, max_hops]
    new_action_logits: torch.Tensor,   # [B, max_hops, 4]
    new_state_values:  torch.Tensor,   # [B, max_hops]
    l_semantic:        torch.Tensor,   # scalar — InfoNCE loss (already computed)
    lambda_sem:        float = 1.0,    # weight of semantic loss
    clip_eps:          float = 0.2,
    entropy_coef:      float = 0.01,
    value_coef:        float = 0.5,
) -> dict:
    """
    Joint PPO + InfoNCE update.

    L_total = λ_sem * L_infonce
            + L_actor          (clipped surrogate)
            + value_coef * L_critic
            - entropy_coef * entropy

    The InfoNCE term is what connects PPO to semantics:
    it forces the backbone to learn WHAT the question is asking for
    (semantic alignment), while PPO learns HOW WIDE the beam should be
    (strategic efficiency).
    """
    dist          = torch.distributions.Categorical(logits=new_action_logits)
    new_log_probs = dist.log_prob(actions)   # [B, max_hops]
    entropy       = dist.entropy().mean()

    ratio  = torch.exp(new_log_probs - old_log_probs)
    surr1  = ratio * advantages
    surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss  = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(new_state_values, returns)

    total_loss = (
        lambda_sem * l_semantic
        + actor_loss
        + value_coef * critic_loss
        - entropy_coef * entropy
    )

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "actor_loss":   actor_loss.item(),
        "critic_loss":  critic_loss.item(),
        "entropy":      entropy.item(),
        "semantic_loss": l_semantic.item(),
        "total_loss":   total_loss.item(),
    }


# ─────────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────────

def train_exp15_strl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp15] Device: {device}")

    # ── Load Mappings ────────────────────────────────────────
    data_dir = os.path.join(ROOT, "data/processed_entity")
    rel2id   = torch.load(os.path.join(data_dir, "relation2id.pt"))
    dom2id   = torch.load(os.path.join(data_dir, "domain2id.pt"))
    id2rel   = {v: k for k, v in rel2id.items()}
    num_rel  = len(rel2id)
    num_dom  = len(dom2id)
    print(f"[Exp15] Vocab: {num_rel} relations, {num_dom} domains")

    # ── Load Exp7 Base Model ─────────────────────────────────
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

    # ── Pre-compute Relation Embedding Bank (frozen teacher) ─
    rel_emb_bank = RelationEmbeddingBank(id2rel, device).to(device)
    rel_emb_bank.eval()
    for p in rel_emb_bank.parameters():
        p.requires_grad = False

    # ── Build STRL Agent ─────────────────────────────────────
    agent = STRLAgent(base_model).to(device)
    print(f"[Exp15] Agent parameters: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")

    # ── Differential Learning Rates ──────────────────────────
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

    # ── Load KG for entity-grounded reward ───────────────────
    print("[Exp15] Loading KG for entity-grounded reward...")
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    kg = torch.load(kg_path, map_location='cpu')
    print(f"[Exp15] KG loaded: {len(kg.get('forward', {}))} forward entities")

    # ── Dataset (KG-anchored) ─────────────────────────────────
    tokenizer  = RobertaTokenizer.from_pretrained("roberta-large")
    train_ds   = STRLDataset(os.path.join(ROOT, "data/cwq_train.json"), rel2id, dom2id, kg, split="train")
    dev_ds     = STRLDataset(os.path.join(ROOT, "data/cwq_dev.json"),   rel2id, dom2id, kg, split="dev")
    collate    = functools.partial(collate_strl, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=8,  shuffle=False, collate_fn=collate)
    print(f"[Exp15] Dataset: {len(train_ds)} train, {len(dev_ds)} dev samples")

    # ── Metrics Setup ─────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "metrics"),      exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"),  exist_ok=True)
    metrics_path = os.path.join(ROOT, "metrics/exp15_strl.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,avg_reward,actor_loss,critic_loss,semantic_loss,entropy,dev_sem_hit1\n")

    # ── Training Config ───────────────────────────────────────
    epochs          = 20
    gamma           = 0.99   # discount factor
    curriculum_end  = 5      # switch to pure semantic reward after epoch 5
    clip_eps        = 0.2
    scaler          = torch.amp.GradScaler("cuda")

    # λ_sem schedule: 1.0 during grounding phase, 0.3 during RL refinement
    # This ensures semantic understanding is established BEFORE RL shapes strategy
    def get_lambda_sem(ep: int) -> float:
        return 1.0 if ep < curriculum_end else 0.3

    # ── Resume Logic ──────────────────────────────────────────
    start_epoch = 0
    best_dev_acc = 0.0
    exp15_ckpts = [f for f in os.listdir(os.path.join(ROOT, 'checkpoints')) 
                   if f.startswith('exp15_strl_epoch_') and f.endswith('.pt')]
    if exp15_ckpts:
        latest_ckpt = max(exp15_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0]) + 1
        ckpt_path = os.path.join(ROOT, 'checkpoints', latest_ckpt)
        print(f"[Exp15] Resuming from {ckpt_path} (Starting Epoch {start_epoch})")
        agent.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # If resuming, we need to find the best_dev_acc from metrics or previous runs
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
        
        # ── Speed Optimization: Partial Freezing ─────────────────
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

        ep_rewards      = []
        ep_actor_loss   = []
        ep_critic_loss  = []
        ep_semantic_loss = []
        ep_entropy      = []
        lambda_sem      = get_lambda_sem(epoch)

        for enc, doms, gold_paths, path_lengths, avail_rels in t_bar:
            enc          = enc.to(device)
            doms         = doms.to(device)
            gold_paths   = gold_paths.to(device)      # [B, max_hops]
            path_lengths = path_lengths.to(device)    # [B]
            # avail_rels stays as list-of-lists (CPU) — used in reward calc only

            # ── Rollout (no grad for sampling) ───────────────
            with torch.no_grad():
                fwd = agent(enc["input_ids"], enc["attention_mask"])

            action_logits = fwd["action_logits"]   # [B, H, 4]
            state_values  = fwd["state_values"]    # [B, H]
            hop_reprs     = fwd["hop_reprs"]       # [B, H, 1024]

            dist    = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()                # [B, H]
            old_log_probs = dist.log_prob(actions) # [B, H]

            # ── Compute Rewards ──────────────────────────────
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

            # ── Compute Returns & Advantages ─────────────────
            B, H = rewards.size()
            returns    = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)

            for b in range(B):
                G = 0.0
                for h in reversed(range(H)):
                    G = rewards[b, h].item() + gamma * G
                    returns[b, h]    = G
                    advantages[b, h] = G - state_values[b, h].detach().item()

            # Normalise advantages (reduces variance)
            adv_mean = advantages.mean()
            adv_std  = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # ── Joint PPO + InfoNCE Update (fresh forward WITH grad) ─
            # This forward pass produces hop_reprs WITH gradient so that
            # InfoNCE can backprop through proj_to_teacher → transformer → RoBERTa
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
                    old_log_probs     = old_log_probs.detach(),
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

        # ── Dev Evaluation: Semantic Hit@1 ───────────────────
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

                actions_dev = fwd["action_logits"].argmax(dim=-1)
                for a in actions_dev.flatten().tolist():
                    action_counts[a] = action_counts.get(a, 0) + 1

                B = gold_paths.size(0)
                for b in range(B):
                    L = int(path_lengths[b].item())
                    for h in range(L):
                        hop_vec  = hop_reprs_dev[b, h]          # [1024]
                        sims     = torch.mv(all_rel_embs, hop_vec)  # [N_rel]
                        pred_rel = sims.argmax().item()         # semantic top-1
                        gold_rel = int(gold_paths[b, h].item())
                        sem_hit1_correct += int(pred_rel == gold_rel)
                        sem_hit1_total   += 1

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

        # ── Metrics Log ──────────────────────────────────────
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_r:.4f},{avg_al:.4f},{avg_cl:.4f},{avg_sl:.4f},{avg_ent:.4f},{dev_acc:.4f}\n")

        # ── Checkpoint ───────────────────────────────────────
        torch.save(agent.state_dict(),
                   os.path.join(ROOT, f"checkpoints/exp15_strl_epoch_{epoch}.pt"))

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(agent.state_dict(),
                       os.path.join(ROOT, "checkpoints/exp15_strl_best.pt"))
            print(f"  * New best Semantic Hit@1: {dev_acc*100:.2f}% - saved exp15_strl_best.pt")

    print("\n[Exp15] Training complete.")
    print(f"  Best dev path accuracy: {best_dev_acc*100:.2f}%")
    print(f"  Best checkpoint: checkpoints/exp15_strl_best.pt")


# ─────────────────────────────────────────────────────────────
#  Inference Helper: Semantic Beam + KG Filter
# ─────────────────────────────────────────────────────────────

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

    Step 1 — Semantic beam:
      Rank all relations by cosine_sim(hop_repr, rel_emb_bank).
      Take top-k according to action (TIGHT=1, MEDIUM=5, LOOSE=50).

    Step 2 — KG intersection:
      Find which of those top-k relations ACTUALLY EXIST on the
      current entity set in the KG.
      If intersection is non-empty → use it (pure semantic + reachable).

    Step 3 — KG-only fallback:
      If semantic beam has ZERO reachable relations (opaque question
      like 'Country Nation World Tour'), fall back to ALL relations
      available on the entity and re-rank those by semantic score.
      This is the fix for blind-shooting on bridge-entity questions.

    Returns: list of (relation_name, score) sorted by score desc
    """
    all_embs = rel_emb_bank.all()                       # [N_rel, 1024]
    sims     = torch.mv(all_embs, hop_repr)             # [N_rel]
    k        = BEAM_SIZES.get(action, 5)
    k        = min(k, all_embs.size(0))
    top_k    = torch.topk(sims, k)
    sem_ids  = top_k.indices.tolist()                   # semantic top-k rel IDs
    sem_sims = top_k.values.tolist()

    # Collect relations actually reachable from current entities
    reachable_rels = set()
    for mid in current_entity_set:
        for rel, _ in kg.get('forward',  {}).get(mid, []):
            if rel in rel2id:
                reachable_rels.add(rel2id[rel])
        for rel, _ in kg.get('backward', {}).get(mid, []):
            if rel in rel2id:
                reachable_rels.add(rel2id[rel])

    id2rel_local = {v: k for k, v in rel2id.items()}

    # Step 2: intersection
    intersection = [(rid, s) for rid, s in zip(sem_ids, sem_sims)
                    if rid in reachable_rels]

    if intersection:
        return [(id2rel_local[rid], s) for rid, s in intersection]

    # Step 3: fallback — rank reachable relations by semantic score
    if reachable_rels:
        fallback = [(rid, sims[rid].item()) for rid in reachable_rels]
        fallback.sort(key=lambda x: x[1], reverse=True)
        return [(id2rel_local[rid], s) for rid, s in fallback[:k]]

    # Nothing reachable at all
    return [(id2rel_local[rid], s) for rid, s in zip(sem_ids, sem_sims)]


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_exp15_strl()
