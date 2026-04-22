import os, sys, json, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.kg_loader import KnowledgeGraph
from utils.sparql_parser import find_reasoning_path, extract_triples

# ============================================================
#  1. Neural Reward Model (Heuristic Scorer for Process-based Reward)
# ============================================================
class TrajectoryRewardModel(nn.Module):
    """
    Scores a (Question, Path_Trajectory) pair.
    In this version, we use it to calculate semantic similarity 
    between the question and the relations chosen.
    """
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :] 
        reward_score = self.score_head(cls_repr)
        return reward_score.squeeze(-1)

# ============================================================
#  2. Generalized Graph Walker Agent (PPO Actor-Critic)
# ============================================================
class GraphWalkerPPOAgent(nn.Module):
    """
    Dynamic Action Space Graph Walker using Semantic Edge Embeddings.
    """
    def __init__(self, edge_embeddings, hidden_dim=512):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-large")
        self.proj = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        
        # Critic computes Value of current Node State (V(s_t))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Action projection to 1024-dim space (match roberta-large embeddings)
        self.action_proj = nn.Linear(hidden_dim, 1024)
        
        # Pre-computed relation embeddings [num_relations, 1024]
        self.register_buffer("edge_embeddings", edge_embeddings)

    def forward(self, input_ids, attention_mask, available_edge_ids):
        """
        available_edge_ids: tensor of indices for relations available at current node
        """
        B = input_ids.size(0)
        outputs = self.encoder(input_ids, attention_mask)
        q_h = outputs.last_hidden_state[:, 0, :]
        state_repr = self.proj(q_h) # [B, hidden_dim]
        
        # 1. Critic Value
        state_value = self.value_head(state_repr).squeeze(-1) # [B]
        
        # 2. Dynamic Actor Probabilities
        if available_edge_ids is None or len(available_edge_ids) == 0:
            return None, state_value, state_repr
            
        target_vectors = self.action_proj(state_repr) # [B, 1024]
        target_vectors = F.normalize(target_vectors, p=2, dim=-1)
        
        # Action Space Construction from available IDs
        # [num_available_edges, 1024]
        action_space = self.edge_embeddings[available_edge_ids]
        action_space = F.normalize(action_space, p=2, dim=-1)
        
        # Cosine Similarity Logits: [B, num_available_edges]
        action_logits = torch.matmul(target_vectors, action_space.T) * 10.0 # Scaling Factor
        
        return action_logits, state_value, state_repr

# ============================================================
#  3. CWQ Environment & Dataset
# ============================================================
class CWQRLDataset(Dataset):
    def __init__(self, json_path, relation2id):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        print(f"Preprocessing {json_path} for RL...")
        for item in tqdm(data[:10000]): # Limit to 10k for faster RL cycles
            path = find_reasoning_path(item['sparql'])
            if path is None: continue
            
            topic_entity = path[0][0].replace('ns:', '')
            gold_answers = {'m.'+ans['answer_id'].replace('m.','') for ans in item.get('answers', []) if 'answer_id' in ans}
            if not gold_answers:
                gold_answers = {path[-1][3].replace('ns:', '')}
                
            self.samples.append({
                'question': item['question'],
                'topic_entity': topic_entity,
                'gold_answers': gold_answers
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class KGEnvironment:
    def __init__(self, kg, relation2id):
        self.kg = kg
        self.relation2id = relation2id

    def get_available_actions(self, current_entity):
        neighbors = self.kg.get_neighbors(current_entity)
        if not neighbors: return []
        
        actions = []
        for rel, direction, tgt in neighbors:
            if rel in self.relation2id:
                actions.append({
                    'rel_id': self.relation2id[rel],
                    'rel_name': rel,
                    'direction': direction,
                    'target': tgt
                })
        return actions

# ============================================================
#  4. PPO Training Loop
# ============================================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0] # terminal value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

def train_advanced_rl_walker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Resources
    print("Loading KG and Embeddings...")
    kg = torch.load(os.path.join(ROOT, 'data/processed_universal/kg.pt'), weights_only=False)
    relation_embeddings = torch.load(os.path.join(ROOT, 'data/processed_universal/relation_embeddings.pt'))
    relation2id = torch.load(os.path.join(ROOT, 'data/processed_universal/relation2id.pt'))
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    dataset = CWQRLDataset(os.path.join(ROOT, 'data/cwq_train.json'), relation2id)
    env = KGEnvironment(kg, relation2id)
    
    agent = GraphWalkerPPOAgent(relation_embeddings).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-6)
    
    max_hops = 3
    epochs = 20
    batch_size = 16 # Small batch for complex RL trajectory sampling
    
    metrics_path = os.path.join(ROOT, "metrics/exp14_rlkgf.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write("epoch,avg_reward,success_rate\n")

    print("\nStarting Advanced RLKGF Training...")
    for epoch in range(epochs):
        agent.train()
        epoch_rewards = []
        successes = 0
        
        # Shuffle dataset for each epoch
        indices = np.random.permutation(len(dataset))
        
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch}"):
            batch_indices = indices[i:i+batch_size]
            batch_samples = [dataset[idx] for idx in batch_indices]
            
            # Policy updates happen over the collected batch
            optimizer.zero_grad()
            batch_loss = 0
            
            for sample in batch_samples:
                # TRAJECTORY COLLECTION
                current_entity = sample['topic_entity']
                gold_answers = sample['gold_answers']
                q_text = sample['question']
                
                encoded = tokenizer(q_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                
                rewards = []
                log_probs = []
                values = []
                entropies = []
                
                for hop in range(max_hops):
                    actions = env.get_available_actions(current_entity)
                    if not actions: 
                        rewards.append(-0.1) # Dead end penalty
                        break
                    
                    # Construct local action space
                    rel_ids = torch.tensor([a['rel_id'] for a in actions], device=device)
                    
                    logits, val, _ = agent(encoded['input_ids'], encoded['attention_mask'], rel_ids)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    
                    action_idx = dist.sample()
                    selected_action = actions[action_idx.item()]
                    
                    log_probs.append(dist.log_prob(action_idx))
                    values.append(val)
                    entropies.append(dist.entropy())
                    
                    # Step environment
                    current_entity = selected_action['target']
                    
                    # Calculate step reward
                    if current_entity in gold_answers:
                        rewards.append(1.0)
                        successes += 1
                        break
                    else:
                        rewards.append(0.0) # Correct hop but not answer yet
                
                if not rewards: continue
                
                # ADVANTAGE COMPUTATION
                val_list = [v.item() for v in values]
                advs = compute_gae(rewards, val_list).to(device)
                returns = advs + torch.tensor(val_list).to(device)
                
                # PPO LOSS (Simplified REINFORCE style for this baseline)
                for lp, adv, v, ret, ent in zip(log_probs, advs, values, returns, entropies):
                    policy_loss = -lp * adv
                    value_loss = F.mse_loss(v, ret)
                    batch_loss += policy_loss + 0.5 * value_loss - 0.01 * ent
            
            if batch_loss != 0:
                batch_loss = batch_loss / len(batch_samples)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                epoch_rewards.append(np.sum(rewards))

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        success_rate = successes / len(dataset)
        print(f"Epoch {epoch} | Avg Reward: {avg_reward:.4f} | Success Rate: {success_rate:.4f}")
        
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_reward:.4f},{success_rate:.4f}\n")
            
        # Save Checkpoint
        torch.save(agent.state_dict(), f"checkpoints/exp14_epoch_{epoch}.pt")

    print("\nTraining Complete.")

if __name__ == '__main__':
    train_advanced_rl_walker()
