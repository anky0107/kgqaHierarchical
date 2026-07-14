"""
Exp 18: Dead-End State Feature + Per-Hop Entity-Grounded Reward
===============================================================

HYPOTHESIS
----------
Exp9's RL state s_t = x_t contains no runtime graph information.
If hop t-1 produced zero entities (dead-end), the policy still acts
as if traversal is going fine, wasting the remaining hops on a
dead trajectory.

Two targeted fixes applied together:

FIX 1 — Dead-end binary feature
  Append a single scalar d_t ∈ {0, 1} to the state:
      s_t = [x_t ; d_t]   ∈ R^{d+1}
  d_t = 1 if the current frontier is empty, 0 otherwise.
  The policy head is widened by 1 input neuron accordingly.
  During training d_t is computed from whether the current beam
  (of size chosen by a_t-1) contains the gold entity.

FIX 2 — Per-hop entity-grounded reward component
  Exp9 rewards per hop:
    TIGHT  +1.0 / -1.0   (gold in top-1 or not)
    MEDIUM +0.5 / -1.0   (gold in top-5 or not)
    LOOSE  +0.1 / -1.0   (domain match or not)
  These check RELATION correctness, not ENTITY reachability.
  We add a small entity-coverage bonus at every hop (not just hop 0):
    +0.3 if gold_entity is reachable via the chosen beam
    -0.2 if the frontier becomes empty (dead-end penalty)
  This directly optimises Reasoning Recall — the metric that is
  our tightest bottleneck at 75.59%.

ARCHITECTURE CHANGES FROM exp9
--------------------------------
  RLConstraintAgent:  policy_head input  512 → 513  (dead-end bit)
                      value_head  input  512 → 513
  Everything else (base model, training loop, A2C loss) unchanged.

EXPECTED GAIN: +1–3% Reasoning Recall, +1–2% Hit@1
CHECKPOINT:    checkpoints/exp18_rl_deadend.pt
METRICS:       metrics/exp18_rl_deadend.csv
"""

import os, sys, functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

import importlib.util, types

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

if "utils.sparql_parser" not in sys.modules:
    stub = types.ModuleType("utils.sparql_parser")
    stub.find_reasoning_path = lambda *a, **kw: None
    sys.modules["utils.sparql_parser"] = stub
if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")

_exp6 = _load_module("train.exp6_unified", os.path.join(ROOT, "train/exp6_unified.py"))
_exp7 = _load_module("train.exp7_roberta", os.path.join(ROOT, "train/exp7_roberta.py"))

UnifiedDataset       = _exp6.UnifiedDataset
collate_unified      = _exp6.collate_unified
ScaledUnifiedPlanner = _exp7.ScaledUnifiedPlanner

# ─────────────────────────────────────────────────────────────
#  Constants  (kept consistent with exp9)
# ─────────────────────────────────────────────────────────────
ACTION_TIGHT  = 0   # top-1
ACTION_MEDIUM = 1   # top-5
ACTION_LOOSE  = 2   # domain fallback (~50 relations)
ACTION_STOP   = 3

BEAM_SIZES = {ACTION_TIGHT: 1, ACTION_MEDIUM: 5, ACTION_LOOSE: 50}

# Entity-coverage bonus/penalty added on top of exp9 relation reward
ENTITY_HIT_BONUS   = +0.3
DEAD_END_PENALTY   = -0.2


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class RLDeadEndAgent(nn.Module):
    """
    Identical to exp9 RLConstraintAgent except:
    - policy_head and value_head accept (hidden_dim + 1) inputs
      to accommodate the dead-end binary feature d_t.
    - Base model parameters are frozen.
    """
    def __init__(self, base_model: ScaledUnifiedPlanner, hidden_dim: int = 512):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        state_dim = hidden_dim + 1      # +1 for dead-end bit

        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, dead_end_flags: torch.Tensor):
        """
        dead_end_flags: [B, max_hops]  float32 tensor of 0/1 values.
        Returns:
            action_logits : [B, max_hops, 4]
            state_values  : [B, max_hops]
            rel_logits    : [B, max_hops, num_rel]
            domain_logits : [B, num_domains]
        """
        with torch.no_grad():
            outputs = self.base_model.encoder(input_ids, attention_mask)
            q_h     = outputs.last_hidden_state[:, 0, :]       # [B, 1024]
            h_q     = self.base_model.proj(q_h)                # [B, 512]

            init_repr   = h_q.unsqueeze(1) + self.base_model.hop_embeddings.unsqueeze(0)
            refined     = self.base_model.transformer(init_repr)   # [B, H, 512]

            rel_logits    = self.base_model.relation_head(refined)     # [B, H, num_rel]
            domain_logits = self.base_model.domain_head(h_q)           # [B, num_domains]

        # Append dead-end bit to state at each hop: [B, H, 513]
        d = dead_end_flags.unsqueeze(-1).to(refined.device)    # [B, H, 1]
        state = torch.cat([refined, d], dim=-1)                # [B, H, 513]

        action_logits = self.policy_head(state)                # [B, H, 4]
        state_values  = self.value_head(state).squeeze(-1)     # [B, H]

        return action_logits, state_values, rel_logits, domain_logits


# ─────────────────────────────────────────────────────────────
#  Dead-end flag computation
# ─────────────────────────────────────────────────────────────

def compute_dead_end_flags(actions: torch.Tensor,
                            rel_logits: torch.Tensor,
                            gold_paths: torch.Tensor,
                            path_lengths: torch.Tensor) -> torch.Tensor:
    """
    Simulate hop-by-hop traversal and set d_t = 1 if the gold relation
    is NOT reachable via the chosen beam at the previous hop
    (i.e. the previous hop was a dead-end for the gold path).

    d_0 = 0 always (topic entity is given, hop 0 is never a dead-end).

    Returns: [B, max_hops] float32 tensor.
    """
    B, max_hops = actions.shape
    flags = torch.zeros(B, max_hops, dtype=torch.float32)

    for b in range(B):
        L = int(path_lengths[b].item())
        prev_dead = False
        for h in range(max_hops):
            flags[b, h] = float(prev_dead)
            if h >= L:
                break
            a      = actions[b, h].item()
            k      = BEAM_SIZES.get(a, 1)
            gold_r = int(gold_paths[b, h].item())
            topk   = torch.topk(rel_logits[b, h], min(k, rel_logits.size(-1))).indices.tolist()
            prev_dead = (gold_r not in topk)

    return flags


# ─────────────────────────────────────────────────────────────
#  Reward
# ─────────────────────────────────────────────────────────────

def calculate_rewards_exp18(actions: torch.Tensor,
                             rel_logits: torch.Tensor,
                             domain_logits: torch.Tensor,
                             gold_paths: torch.Tensor,
                             gold_domains: torch.Tensor,
                             path_lengths: torch.Tensor) -> torch.Tensor:
    """
    Exp9 base reward + entity-coverage bonus + dead-end penalty.

    Entity-coverage bonus:
      At each valid hop h, check whether gold relation is reachable
      via the chosen beam width:
        gold reachable → +ENTITY_HIT_BONUS
        frontier empty (prev dead-end implied by flags) → +DEAD_END_PENALTY
    """
    B, max_hops = actions.shape
    rewards = torch.zeros(B, max_hops, device=actions.device)

    for b in range(B):
        L        = int(path_lengths[b].item())
        dom      = int(gold_domains[b].item())
        pred_dom = torch.argmax(domain_logits[b]).item()

        for h in range(max_hops):
            a = actions[b, h].item()

            # ── Past true path length: only STOP is correct ───────────────
            if h >= L:
                rewards[b, h] = +1.0 if a == ACTION_STOP else -1.0
                continue

            gold_r  = int(gold_paths[b, h].item())
            logits_h = rel_logits[b, h]

            # ── Early stop ────────────────────────────────────────────────
            if a == ACTION_STOP:
                rewards[b, h] = -1.0
                continue

            # ── Relation-level reward (same as exp9) ──────────────────────
            if a == ACTION_TIGHT:
                top1 = torch.argmax(logits_h).item()
                rel_reward = +1.0 if top1 == gold_r else -1.0

            elif a == ACTION_MEDIUM:
                top5 = torch.topk(logits_h, 5).indices.tolist()
                rel_reward = +0.5 if gold_r in top5 else -1.0

            else:  # ACTION_LOOSE
                rel_reward = +0.1 if pred_dom == dom else -1.0

            # ── Entity-coverage bonus (NEW) ───────────────────────────────
            k    = BEAM_SIZES.get(a, 1)
            topk = torch.topk(logits_h, min(k, logits_h.size(-1))).indices.tolist()
            if gold_r in topk:
                entity_bonus = ENTITY_HIT_BONUS
            else:
                entity_bonus = DEAD_END_PENALTY      # will be empty frontier

            rewards[b, h] = rel_reward + entity_bonus

    return rewards


# ─────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────

def train_exp18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp18] Device: {device}")

    rel2id = torch.load(os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    dom2id = torch.load(os.path.join(ROOT, "data/processed_entity/domain2id.pt"))
    num_rel = len(rel2id); num_dom = len(dom2id)

    # ── Base model (exp7 checkpoint) ──────────────────────────────────────────
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base_ckpt  = os.path.join(ROOT, "checkpoints/exp7_roberta_best.pt")
    base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
    print(f"[Exp18] Loaded base model from {base_ckpt}")

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent     = RLDeadEndAgent(base_model).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4)

    tokenizer    = RobertaTokenizer.from_pretrained("roberta-large")
    collate      = functools.partial(collate_unified, tokenizer=tokenizer)
    train_ds     = UnifiedDataset("data/cwq_train.json", rel2id, dom2id)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)

    epochs = 10
    gamma  = 0.99

    metrics_dir  = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp18_rl_deadend.csv")
    with open(metrics_path, "w") as f:
        f.write("epoch,avg_reward,avg_loss\n")

    print("\n[Exp18] Training — Dead-End State Feature + Entity-Coverage Reward")

    for epoch in range(epochs):
        agent.train()
        total_reward = 0.0; total_loss = 0.0; n_steps = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs-1}")

        for enc, doms, paths, nums in pbar:
            enc   = enc.to(device)
            doms  = doms.to(device)
            paths = paths.to(device)
            nums  = nums.to(device)

            with torch.amp.autocast("cuda"):
                # ── Pass 1: get rel_logits with zero dead-end flags ───────
                B, H = paths.shape
                zero_flags = torch.zeros(B, H, device=device)
                action_logits, state_values, rel_logits, domain_logits = \
                    agent(enc["input_ids"], enc["attention_mask"], zero_flags)

                # ── Sample actions ────────────────────────────────────────
                probs   = F.softmax(action_logits, dim=-1)
                dist    = torch.distributions.Categorical(probs)
                actions = dist.sample()           # [B, H]
                log_probs = dist.log_prob(actions) # [B, H]

                # ── Compute dead-end flags from sampled actions ───────────
                dead_flags = compute_dead_end_flags(
                    actions.cpu(), rel_logits.detach().cpu(),
                    paths.cpu(), nums.cpu()).to(device)

                # ── Pass 2: forward with actual dead-end flags ────────────
                action_logits2, state_values2, _, _ = \
                    agent(enc["input_ids"], enc["attention_mask"], dead_flags)

                probs2    = F.softmax(action_logits2, dim=-1)
                dist2     = torch.distributions.Categorical(probs2)
                log_probs2 = dist2.log_prob(actions)

                # ── Rewards ───────────────────────────────────────────────
                rewards = calculate_rewards_exp18(
                    actions, rel_logits.detach(), domain_logits.detach(),
                    paths, doms, nums)
                total_reward += rewards.mean().item()

                # ── Advantages (A2C, same as exp9) ────────────────────────
                returns = torch.zeros_like(rewards)
                adv     = torch.zeros_like(rewards)
                for b in range(B):
                    G = 0.0
                    for h in reversed(range(H)):
                        G = rewards[b, h].item() + gamma * G
                        returns[b, h] = G
                        adv[b, h]     = G - state_values2[b, h].item()

                returns = returns.to(device)
                adv     = adv.to(device)

                actor_loss  = -(log_probs2 * adv).mean()
                critic_loss = F.mse_loss(state_values2, returns)
                entropy     = -dist2.entropy().mean() * 0.01
                loss        = actor_loss + 0.5 * critic_loss + entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps    += 1
            pbar.set_postfix(
                reward=f"{rewards.mean().item():.3f}",
                loss=f"{loss.item():.4f}")

        avg_r = total_reward / n_steps
        avg_l = total_loss   / n_steps
        print(f"Epoch {epoch} | Avg Reward: {avg_r:.4f} | Avg Loss: {avg_l:.4f}")
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_r:.4f},{avg_l:.4f}\n")

        ckpt = os.path.join(ROOT, f"checkpoints/exp18_rl_deadend_epoch_{epoch}.pt")
        torch.save(agent.state_dict(), ckpt)

    print("[Exp18] Training complete.")


if __name__ == "__main__":
    train_exp18()
