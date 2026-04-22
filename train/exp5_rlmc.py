# train/exp5_rlmc.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.encoder import QuestionEncoder

# Action Space Definition
# 0: TIGHT (top-1)
# 1: MEDIUM (top-5)
# 2: LOOSE (domain)
# 3: STOP
NUM_ACTIONS = 4

class RLMCPolicy(nn.Module):
    def __init__(self, question_dim=768, entity_dim=128, path_dim=128, hidden_dim=256):
        super().__init__()
        
        # We assume the question is already encoded by the BERT QuestionEncoder
        # entity_emb is retrieved from freebase pre-trained embeddings
        
        state_dim = question_dim + entity_dim + path_dim + 2 # +2 for hop_number and candidate_count
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, NUM_ACTIONS)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # state: [B, state_dim]
        h = self.net(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        
        probs = F.softmax(logits, dim=-1)
        return probs, value, logits

class RLMCEnvironment:
    def __init__(self, kg_loader, relation_scorer):
        self.kg = kg_loader
        self.scorer = relation_scorer
        self.max_hops = 3
        
    def reset(self, question_text, start_entity, gold_answer):
        # Initialize state components
        self.current_entity = start_entity
        self.question = question_text
        self.gold_answer = gold_answer
        self.hop = 0
        self.path_so_far = []
        # Return initial state tensor (dummy for now)
        return self._get_state()
        
    def _get_state(self):
        # Constructs state tensor: [question_emb, entity_emb, hop_number, path_so_far_emb, candidate_count]
        # In actual implementation, requires accessing BERT embeddings and graph embeddings
        return torch.randn(1, 768 + 128 + 128 + 2)
        
    def step(self, action):
        # Apply action
        # 0: Tight -> top 1 relation
        # 1: Medium -> top 5 relations
        # ...
        reward = 0.0
        done = False
        
        if action == 3 or self.hop >= self.max_hops:
            done = True
            if self.current_entity == self.gold_answer:
                reward += 1.0 # r_final
            return self._get_state(), reward, done
            
        # Mock traversal
        candidates = self.kg.get_neighbors(self.current_entity)
        
        # Efficiency penalty
        reward -= 0.1 * len(candidates)
        
        # Transition ...
        self.hop += 1
        
        return self._get_state(), reward, done

def compute_ppo_loss(old_probs, new_probs, advantages, returns, values, epsilon=0.2, c1=0.5, c2=0.01):
    ratio = new_probs / (old_probs + 1e-8)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(values, returns)
    
    # Optional entropy bonus for exploration
    entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(-1).mean()
    
    loss = actor_loss + c1 * critic_loss - c2 * entropy
    return loss

def train_ppo_mock():
    # Mock training loop to demonstrate PPO integration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
    policy = RLMCPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Collect rollouts
    batch_states = torch.randn(64, 1026).to(device)
    batch_actions = torch.randint(0, 4, (64,)).to(device)
    batch_advantages = torch.randn(64).to(device)
    batch_returns = torch.randn(64).to(device)
    batch_old_probs = torch.rand(64).to(device)
    
    # Update policy
    for epoch in range(4): # PPO epochs
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                probs, values, logits = policy(batch_states)
                action_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                loss = compute_ppo_loss(batch_old_probs, action_probs, batch_advantages, batch_returns, values)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            probs, values, logits = policy(batch_states)
            action_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            loss = compute_ppo_loss(batch_old_probs, action_probs, batch_advantages, batch_returns, values)
            loss.backward()
            optimizer.step()
        print(f"PPO Epoch {epoch} complete. Loss: {loss.item():.4f}")
        
    print("PPO Step complete.")

if __name__ == "__main__":
    train_ppo_mock()
