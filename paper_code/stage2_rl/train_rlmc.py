"""
train_rlmc.py — Stage 2: RL Meta-Constraint Agent (RLMC) — Experiment 9
========================================================================

PURPOSE
-------
This file implements the first RL-based fine-tuning stage after the supervised
Stage 1 planner.  A lightweight **RL Meta-Constraint Agent (RLMC)** learns to
select the optimal *constraint action* for each reasoning hop — one of:

  ACTION 0 — TIGHT  : use only the top-1 predicted relation  (high precision)
  ACTION 1 — MEDIUM : use top-5 predicted relations           (balanced)
  ACTION 2 — LOOSE  : fall back to domain-wide relations (~50) (high recall)
  ACTION 3 — STOP   : terminate path traversal at this hop

The policy is trained with a simplified PPO / A2C actor-critic algorithm.
The reward signal is derived by checking whether the chosen constraint width
"captures" the gold relation ID — no semantic teacher is required at this stage.

PAPER SECTION
-------------
Corresponds to **Experiment 9** ("RL Meta-Constraint Agent") in the paper.
This ablates the frozen-backbone approach: the Stage 1 RoBERTa-based planner
(Exp 7) is fully **frozen**, and only the policy/value heads are trained with RL.

Key difference from Exp 15 (train_strl.py):
  - Exp 9  : backbone FROZEN, reward = gold relation ID hard-match
  - Exp 15 : backbone UNFROZEN (joint), reward = SEMANTIC cosine similarity

PIPELINE POSITION
-----------------
  [Exp 7 checkpoint: exp7_roberta_best.pt]  -- frozen RoBERTa planner
           |
           v
  RLConstraintAgent: adds 4-action policy head + value head on top of frozen base
           |
           v
  PPO/A2C RL training using gold-path rewards  ->  exp9_rlmc_epoch_{e}.pt

INPUTS
------
  checkpoints/exp7_roberta_best.pt         : pre-trained Exp 7 backbone weights
  data/cwq_train.json                      : CWQ training split
  data/cwq_dev.json                        : CWQ dev split
  data/processed_entity/relation2id.pt     : dict[rel_str -> int]
  data/processed_entity/domain2id.pt       : dict[domain_str -> int]

OUTPUTS
-------
  checkpoints/exp9_rlmc_epoch_{e}.pt : checkpoint saved each epoch
  metrics/exp9_rlmc.csv              : per-epoch average reward

KEY HYPERPARAMETERS
-------------------
  hidden_dim   = 512          (Exp 7 uses 512; policy/value heads use 256)
  num_actions  = 4            (TIGHT / MEDIUM / LOOSE / STOP)
  batch_size   = 8 (train), 16 (dev)
  lr           = 1e-4         (higher than Exp 6 because base is frozen)
  gamma        = 0.99         (discount factor for returns)
  entropy_coef = 0.01         (exploration bonus coefficient)
  value_coef   = 0.5          (critic loss weight in combined loss)
  epochs       = 10

HOW IT WORKS
------------
  1. The frozen Exp 7 backbone encodes the question and produces refined hop
     representations [B, max_hops, 512] and relation logits [B, max_hops, R].
  2. The policy head maps hop representations -> action logits [B, max_hops, 4].
  3. Actions are sampled from Categorical(softmax(action_logits)).
  4. Rewards are computed per-hop by `calculate_meta_rewards`:
       - Past true length: STOP gets +1.0, others -1.0
       - TIGHT: +1.0 if top-1 prediction matches gold, else -1.0
       - MEDIUM: +0.5 if gold in top-5, else -1.0
       - LOOSE:  +0.1 if predicted domain matches gold, else -1.0
       - STOP (early): -1.0
  5. Returns are computed via discounted cumulative reward (backward pass).
  6. Advantage = Return - baseline (current state value estimate).
  7. Combined loss = actor_loss + 0.5 * critic_loss + entropy_bonus
     where entropy_bonus encourages exploration (prevents premature collapse).
"""

import os, sys, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

# ── Resolve project root so shared modules can be imported regardless of CWD ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train.exp6_unified import UnifiedDataset, collate_unified
from train.exp7_roberta import ScaledUnifiedPlanner


# ──────────────────────────────────────────────────────────────────────────────
#  RL Agent Definition
# ──────────────────────────────────────────────────────────────────────────────

class RLConstraintAgent(nn.Module):
    """
    Wraps the frozen Exp 7 backbone with learnable RL policy and value heads.

    The frozen base model provides:
      - Per-hop relation logits  [B, max_hops, num_rel]
      - Domain logits            [B, num_domains]
      - Refined hop reprs        [B, max_hops, 512]  (intermediate activations)

    The new RL heads take the refined hop representations as input and output:
      - action_logits : [B, max_hops, 4]   -> which constraint width to apply
      - state_values  : [B, max_hops]      -> estimated return from this state
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Freeze ALL base model parameters — only RL heads will receive gradients
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_dim = 512   # matches Exp 7 hidden dimension
        # The Action Policy Head: outputs probabilities for 4 constraint actions
        # 0: TIGHT (top-1)
        # 1: MEDIUM (top-5)
        # 2: LOOSE (domain fallback)
        # 3: STOP
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        # Value head for PPO advantage calculation: estimates expected return
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids      : [B, seq_len]
            attention_mask : [B, seq_len]

        Returns:
            action_logits  : [B, max_hops, 4]   raw policy scores per action
            state_values   : [B, max_hops]       value estimates per hop
            rel_logits     : [B, max_hops, R]    frozen base relation logits
            domain_logits  : [B, num_domains]    frozen base domain logits
        """
        # ── Step 1: Run frozen backbone (no gradient flows through here) ──────
        with torch.no_grad():
            B = input_ids.size(0)
            # RoBERTa encoder; use CLS token as question representation
            outputs = self.base_model.encoder(input_ids, attention_mask)
            q_h = outputs.last_hidden_state[:, 0, :]  # [B, 1024] RoBERTa CLS
            h_q = self.base_model.proj(q_h)           # [B, 512] projected repr

            # Reproduce the Exp 7 cross-hop reasoning: add hop positional embeddings
            # then run Transformer to get coherent per-hop representations
            init_repr    = h_q.unsqueeze(1) + self.base_model.hop_embeddings.unsqueeze(0)
            refined_repr = self.base_model.transformer(init_repr)  # [B, max_hops, 512]

            # rel_logits shape: [B, max_hops, num_rel]
            rel_logits    = self.base_model.relation_head(refined_repr)
            domain_logits = self.base_model.domain_head(h_q)

        # ── Step 2: RL heads receive refined hop representations ──────────────
        # RL Heads take the refined hop representations and output policy actions
        # refined_repr shape: [B, max_hops, hidden_dim]
        action_logits = self.policy_head(refined_repr) # [B, 4, 4]
        state_values  = self.value_head(refined_repr).squeeze(-1) # [B, 4]

        return action_logits, state_values, rel_logits, domain_logits


# ──────────────────────────────────────────────────────────────────────────────
#  Reward Function
# ──────────────────────────────────────────────────────────────────────────────

def calculate_meta_rewards(actions, rel_logits, domain_logits, gold_paths, gold_domains, path_lengths):
    """
    Rewards the RL agent based on efficiency vs accuracy trade-off.

    The reward is computed per (batch, hop) pair and reflects whether the
    chosen constraint action would correctly capture the gold relation:

      STOP (past true length)  -> +1.0  (correct termination)
      non-STOP (past length)   -> -1.0  (should have stopped)
      STOP (within length)     -> -1.0  (stopped too early)
      TIGHT + gold in top-1   -> +1.0  (precise and correct)
      TIGHT + gold not top-1  -> -1.0  (beam too narrow)
      MEDIUM + gold in top-5  -> +0.5  (correct but less efficient)
      MEDIUM + gold not top-5 -> -1.0
      LOOSE + domain matches  -> +0.1  (correct but very inefficient)
      LOOSE + domain mismatch -> -1.0

    The declining reward (+1 > +0.5 > +0.1) incentivises the agent to choose
    the tightest beam that still covers the gold relation.

    Args:
        actions        : [B, max_hops] int64 action indices
        rel_logits     : [B, max_hops, num_rel] frozen relation logits
        domain_logits  : [B, num_domains] frozen domain logits
        gold_paths     : [B, max_hops] gold relation IDs (0-padded)
        gold_domains   : [B] gold domain IDs
        path_lengths   : [B] number of valid hops per sample

    Returns:
        rewards        : [B, max_hops] float tensor
    """
    B, max_hops = actions.size()
    rewards = torch.zeros(B, max_hops).to(actions.device)

    for b in range(B):
        L        = int(path_lengths[b].item())   # true path length for sample b
        dom      = int(gold_domains[b].item())   # gold domain index
        pred_dom = torch.argmax(domain_logits[b]).item()  # predicted domain (argmax)

        for h in range(max_hops):
            a = actions[b, h].item()  # scalar action chosen by the policy

            # ── Past true path length: only STOP is correct ───────────────────
            if h >= L:
                if a == 3: # STOP
                    rewards[b, h] = +1.0
                else:
                    rewards[b, h] = -1.0   # continued when should have stopped
                continue

            # ── Within the true path: evaluate constraint accuracy ─────────────
            gold_r  = int(gold_paths[b, h].item())  # gold relation ID this hop
            logits_h = rel_logits[b, h]              # [num_rel] frozen relation logits

            if a == 3: # STOP early
                rewards[b, h] = -1.0 # Failed to reach answer

            elif a == 0: # TIGHT (Top-1) — highest precision, highest risk
                top1 = torch.argmax(logits_h).item()
                if top1 == gold_r:
                    rewards[b, h] = +1.0 # High efficiency, correct!
                else:
                    rewards[b, h] = -1.0 # Failed, beam too tight

            elif a == 1: # MEDIUM (Top-5) — balanced recall vs precision
                top5 = torch.topk(logits_h, 5).indices.tolist()
                if gold_r in top5:
                    rewards[b, h] = +0.5 # Correct, but less efficient explore
                else:
                    rewards[b, h] = -1.0 # Failed anyway

            elif a == 2: # LOOSE (Domain logic) — broadest constraint
                # If predicted domain matches gold domain, it theoretically contains the relation
                if pred_dom == dom:
                    rewards[b, h] = +0.1 # Correct, but highly inefficient (searching 50+ rels)
                else:
                    rewards[b, h] = -1.0

    return rewards


# ──────────────────────────────────────────────────────────────────────────────
#  Training Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def train_exp9_rlmc():
    """
    Training loop for the RL Meta-Constraint Agent (Exp 9).

    Uses a simplified on-policy PPO / A2C update:
      1. Forward pass through frozen base + learnable RL heads.
      2. Sample actions from policy distribution.
      3. Compute per-hop rewards via `calculate_meta_rewards`.
      4. Compute discounted returns and advantages (Return - V(s)).
      5. Actor loss = -E[log_prob * advantage]          (REINFORCE baseline)
         Critic loss = MSE(state_values, returns)       (value function regression)
         Entropy bonus = -0.01 * E[H(pi)]               (exploration encouragement)
      6. Checkpoint saved every epoch (no dev metric gating in this experiment).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load vocabulary maps ──────────────────────────────────────────────────
    rel2id  = torch.load('data/processed_entity/relation2id.pt')
    dom2id  = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id); num_dom = len(dom2id)

    # ── Load frozen Exp 7 base model ──────────────────────────────────────────
    # Base Model: load pre-trained RoBERTa-Large planner from Exp 7
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base_model.load_state_dict(
        torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device)
    )

    # ── Wrap base with RL heads ───────────────────────────────────────────────
    # RL Agent: only policy_head + value_head have requires_grad=True
    rl_agent  = RLConstraintAgent(base_model).to(device)
    # Higher LR than Stage 1 because only the small RL heads are being trained
    optimizer = torch.optim.AdamW(rl_agent.parameters(), lr=1e-4) # Higher LR because base is frozen

    # ── Dataset (reuses Exp 6 dataset class with RoBERTa tokeniser) ──────────
    tokenizer    = RobertaTokenizer.from_pretrained('roberta-large')
    train_ds     = UnifiedDataset('data/cwq_train.json', rel2id, dom2id)
    dev_ds       = UnifiedDataset('data/cwq_dev.json',   rel2id, dom2id)
    collate      = functools.partial(collate_unified, tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=16,                collate_fn=collate)

    # ── Training hyper-parameters ─────────────────────────────────────────────
    epochs = 10
    gamma  = 0.99   # discount factor for computing returns

    # ── Metrics initialisation ────────────────────────────────────────────────
    metrics_path = os.path.join(ROOT, "metrics", "exp9_rlmc.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,avg_reward\n")

    print(f"\nStarting Fast PPO Meta-Constraint Training (Exp 9)...")

    for epoch in range(epochs):
        rl_agent.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_reward_epoch = 0

        for enc, doms, paths, nums in t_bar:
            enc = enc.to(device); doms = doms.to(device); paths = paths.to(device); nums = nums.to(device)

            with torch.amp.autocast('cuda'):
                # ── Step 1: Forward Pass ──────────────────────────────────────
                # Get action logits, value estimates, and frozen base outputs
                action_logits, state_values, rel_logits, domain_logits = rl_agent(enc['input_ids'], enc['attention_mask'])

                # ── Step 2: Sample Actions (Categorical) ──────────────────────
                # Convert logits to probabilities; sample a discrete action per hop
                probs = F.softmax(action_logits, dim=-1)
                m     = torch.distributions.Categorical(probs)
                actions    = m.sample()      # [B, 4]  sampled action per hop
                log_probs  = m.log_prob(actions) # [B, 4]  log-prob of chosen action

                # ── Step 3: Calculate Meta-Rewards ────────────────────────────
                # Reward based on whether chosen beam width captures the gold relation
                rewards = calculate_meta_rewards(actions, rel_logits, domain_logits, paths, doms, nums)
                total_reward_epoch += rewards.mean().item()

                # ── Step 4: Compute Advantages and Returns ────────────────────
                # Discounted return G_t = r_t + gamma * G_{t+1}
                # Advantage A_t = G_t - V(s_t)  (baseline subtraction reduces variance)
                returns = torch.zeros_like(rewards)
                adv     = torch.zeros_like(rewards)

                B, H = rewards.size()
                for b in range(B):
                    G = 0
                    for h in reversed(range(H)):   # backward sweep for discounting
                        G           = rewards[b, h] + gamma * G
                        returns[b, h] = G
                        adv[b, h]   = G - state_values[b, h].item()

                # ── Step 5: Actor-Critic Loss (Simplified PPO / A2C) ──────────
                # Actor loss: maximise E[log_prob * advantage] (gradient ascent on reward)
                actor_loss = -(log_probs * adv).mean()
                # Critic loss: regress state values to actual discounted returns
                critic_loss = F.mse_loss(state_values, returns)
                # Entropy bonus: encourages exploration by penalising a peaked distribution
                entropy_bonus = -m.entropy().mean() * 0.01 # Explore!

                # Combined loss (value_coef=0.5 down-weights critic relative to actor)
                loss = actor_loss + 0.5 * critic_loss + entropy_bonus

            # Standard backward + step (no AMP scaler since base is frozen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_bar.set_postfix(reward=rewards.mean().item(), loss=loss.item())

        # ── Epoch summary and checkpoint ──────────────────────────────────────
        avg_r = total_reward_epoch / len(train_loader)
        print(f"Epoch {epoch} | Avg Meta-Reward: {avg_r:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_r:.4f}\n")

        # Save every epoch (no dev gating — reward is the primary signal here)
        torch.save(rl_agent.state_dict(), f'checkpoints/exp9_rlmc_epoch_{epoch}.pt')


if __name__ == "__main__":
    train_exp9_rlmc()
