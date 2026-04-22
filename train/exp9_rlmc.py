"""
Exp 9: RL Meta-Constraint Agent (RLMC)

Trains an RL policy over 4 constraint actions: [TIGHT, MEDIUM, LOOSE, STOP].
We freeze the RoBERTa sequence planner and train a 4-action PPO head.
Reward is calculated instantaneously by checking if the chosen constraint 
width successfully captures the gold relation for that hop, penalizing for inefficiency.
"""
import os, sys, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp6_unified import UnifiedDataset, collate_unified
from train.exp7_roberta import ScaledUnifiedPlanner

class RLConstraintAgent(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        hidden_dim = 512
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
        # Value head for PPO advantage calculation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            B = input_ids.size(0)
            outputs = self.base_model.encoder(input_ids, attention_mask)
            q_h = outputs.last_hidden_state[:, 0, :]
            h_q = self.base_model.proj(q_h)
            
            init_repr = h_q.unsqueeze(1) + self.base_model.hop_embeddings.unsqueeze(0)
            refined_repr = self.base_model.transformer(init_repr)
            
            # rel_logits shape: [B, max_hops, num_rel]
            rel_logits = self.base_model.relation_head(refined_repr)
            domain_logits = self.base_model.domain_head(h_q)
            
        # RL Heads take the refined hop representations and output policy actions
        # refined_repr shape: [B, max_hops, hidden_dim]
        action_logits = self.policy_head(refined_repr) # [B, 4, 4]
        state_values = self.value_head(refined_repr).squeeze(-1) # [B, 4]
        
        return action_logits, state_values, rel_logits, domain_logits

def calculate_meta_rewards(actions, rel_logits, domain_logits, gold_paths, gold_domains, path_lengths):
    """
    Rewards the RL agent based on efficiency vs accuracy trade-off.
    actions: [B, 4] containing integers 0-3
    """
    B, max_hops = actions.size()
    rewards = torch.zeros(B, max_hops).to(actions.device)
    
    for b in range(B):
        L = int(path_lengths[b].item())
        dom = int(gold_domains[b].item())
        pred_dom = torch.argmax(domain_logits[b]).item()
        
        for h in range(max_hops):
            a = actions[b, h].item()
            
            # If we are past the true length, only STOP is correct
            if h >= L:
                if a == 3: # STOP
                    rewards[b, h] = +1.0
                else:
                    rewards[b, h] = -1.0
                continue
                
            # If we are within the true length
            gold_r = int(gold_paths[b, h].item())
            logits_h = rel_logits[b, h]
            
            if a == 3: # STOP early
                rewards[b, h] = -1.0 # Failed to reach answer
            
            elif a == 0: # TIGHT (Top-1)
                top1 = torch.argmax(logits_h).item()
                if top1 == gold_r:
                    rewards[b, h] = +1.0 # High efficiency, correct!
                else:
                    rewards[b, h] = -1.0 # Failed, beam too tight
            
            elif a == 1: # MEDIUM (Top-5)
                top5 = torch.topk(logits_h, 5).indices.tolist()
                if gold_r in top5:
                    rewards[b, h] = +0.5 # Correct, but less efficient explore
                else:
                    rewards[b, h] = -1.0 # Failed anyway
            
            elif a == 2: # LOOSE (Domain logic)
                # If predicted domain matches gold domain, it theoretically contains the relation
                if pred_dom == dom:
                    rewards[b, h] = +0.1 # Correct, but highly inefficient (searching 50+ rels)
                else:
                    rewards[b, h] = -1.0
                    
    return rewards

def train_exp9_rlmc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    rel2id = torch.load('data/processed_entity/relation2id.pt')
    dom2id = torch.load('data/processed_entity/domain2id.pt')
    num_rel = len(rel2id); num_dom = len(dom2id)
    
    # Base Model
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base_model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    
    # RL Agent
    rl_agent = RLConstraintAgent(base_model).to(device)
    optimizer = torch.optim.AdamW(rl_agent.parameters(), lr=1e-4) # Higher LR because base is frozen
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    train_ds = UnifiedDataset('data/cwq_train.json', rel2id, dom2id)
    dev_ds = UnifiedDataset('data/cwq_dev.json', rel2id, dom2id)
    collate = functools.partial(collate_unified, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=16, collate_fn=collate)
    
    epochs = 10
    gamma = 0.99
    
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
                # 1. Forward Pass
                action_logits, state_values, rel_logits, domain_logits = rl_agent(enc['input_ids'], enc['attention_mask'])
                
                # 2. Sample Actions (Categorical)
                probs = F.softmax(action_logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                actions = m.sample() # [B, 4]
                log_probs = m.log_prob(actions) # [B, 4]
                
                # 3. Calculate Meta-Rewards
                rewards = calculate_meta_rewards(actions, rel_logits, domain_logits, paths, doms, nums)
                total_reward_epoch += rewards.mean().item()
                
                # 4. Compute Advantages and Returns
                returns = torch.zeros_like(rewards)
                adv = torch.zeros_like(rewards)
                
                B, H = rewards.size()
                for b in range(B):
                    G = 0
                    for h in reversed(range(H)):
                        G = rewards[b, h] + gamma * G
                        returns[b, h] = G
                        adv[b, h] = G - state_values[b, h].item()
                
                # 5. Actor-Critic Loss (Simplified PPO / A2C)
                actor_loss = -(log_probs * adv).mean()
                critic_loss = F.mse_loss(state_values, returns)
                entropy_bonus = -m.entropy().mean() * 0.01 # Explore!
                
                loss = actor_loss + 0.5 * critic_loss + entropy_bonus
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_bar.set_postfix(reward=rewards.mean().item(), loss=loss.item())
            
        avg_r = total_reward_epoch / len(train_loader)
        print(f"Epoch {epoch} | Avg Meta-Reward: {avg_r:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_r:.4f}\n")
            
        torch.save(rl_agent.state_dict(), f'checkpoints/exp9_rlmc_epoch_{epoch}.pt')

if __name__ == "__main__":
    train_exp9_rlmc()
