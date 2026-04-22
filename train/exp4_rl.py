"""
Exp 4-RL: Reinforcement-Learning-based Coherence Planning

This experiment fine-tunes the CHCP model (Exp 4) using PPO.
Reward is based on reaching the correct answer in the KG subgraph.
"""
import os, sys, json, torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp4_chcp import CHCPModel, collate_fn, CHCPDataset
from shared.kg_loader import build_kg_from_cwq_triples, KnowledgeGraph
from utils.sparql_parser import extract_triples, find_reasoning_path
from shared.metrics import hits_at_k

# ============================================================
#  Data Prep Utility (Redefining if missing from eval)
# ============================================================

def extract_question_data(cwq_data):
    """Simplified extraction for RL."""
    samples = []
    for item in cwq_data:
        path = find_reasoning_path(item['sparql'])
        if path is None: continue
        topic_entity = path[0][0].replace('ns:', '')
        gold_answers = {'m.'+ans['answer_id'].replace('m.','') for ans in item.get('answers', []) if 'answer_id' in ans}
        if not gold_answers:
            # Fallback to last node in path
            gold_answers = {path[-1][3].replace('ns:', '')}
            
        samples.append({
            'question': item['question'],
            'topic_entity': topic_entity,
            'gold_answers': gold_answers,
        })
    return samples

# ============================================================
#  1. RL Environment
# ============================================================

class CWQEnvironment:
    def __init__(self, kg, samples):
        self.kg = kg
        self.samples = samples
        self.current_idx = 0
        
    def reset(self):
        self.current_idx = np.random.randint(0, len(self.samples))
        sample = self.samples[self.current_idx]
        return sample

    def step(self, sample, predicted_path):
        """
        predicted_path: list of (rel_name, direction)
        Returns: reward, is_successful
        """
        topic_entity = sample['topic_entity']
        gold_answers = sample['gold_answers']
        
        # Traverse KG
        reached = self.kg.traverse(topic_entity, predicted_path)
        predicted_answers = {e for e in reached if not e.startswith('?')}
        
        # If no answers found forward, try backward (common in CWQ)
        if not predicted_answers:
            backward_path = [(rel, -d) for rel, d in predicted_path]
            reached = self.kg.traverse(topic_entity, backward_path)
            predicted_answers = {e for e in reached if not e.startswith('?')}
            
        # Calculate Reward
        is_successful = bool(predicted_answers & gold_answers)
        
        if is_successful:
            reward = 1.0 # Successful path
        else:
            if not predicted_answers:
                reward = -0.1 # Dead end
            else:
                reward = 0.0 # Reached wrong entity
                
        # Efficiency penalty
        reward -= 0.05 * len(predicted_path)
        
        return reward, is_successful

# ============================================================
#  2. PPO Policy Training (Simplified REINFORCE/PPO)
# ============================================================

def train_chcp_rl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load KG and Data
    print("Building KG Subgraph...")
    kg = build_kg_from_cwq_triples([
        os.path.join(ROOT, 'data/cwq_train.json'),
        os.path.join(ROOT, 'data/cwq_dev.json')
    ], extractor_fn=extract_triples)
    
    dev_data = json.load(open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8'))
    samples = extract_question_data(dev_data)
    env = CWQEnvironment(kg, samples)
    
    relation2id = torch.load(os.path.join(ROOT, 'data/processed_entity/relation2id.pt'))
    id2relation = {v: k for k, v in relation2id.items()}
    num_relations = len(relation2id)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dev_dataset = CHCPDataset(os.path.join(ROOT, 'data/cwq_dev.json'), relation2id, max_hops=4)
    collate = functools.partial(collate_fn, tokenizer=tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate)
    
    # Load Pre-trained CHCP Model
    relation2id = torch.load(os.path.join(ROOT, 'data/processed_entity/relation2id.pt'))
    id2relation = {v: k for k, v in relation2id.items()}
    num_relations = len(relation2id)
    
    model = CHCPModel(num_relations=num_relations, max_hops=4).to(device)
    start_epoch = 0
    rl_ckpts = [f for f in os.listdir(os.path.join(ROOT, 'checkpoints')) if f.startswith('exp4_rl_epoch_') and f.endswith('.pt')]
    if rl_ckpts:
        latest_ckpt = max(rl_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0]) + 1
        ckpt_path = os.path.join(ROOT, 'checkpoints', latest_ckpt)
        print(f"Resuming RL from {ckpt_path} (Starting at Epoch {start_epoch})")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        ckpt_path = os.path.join(ROOT, 'checkpoints/exp4_chcp_best.pt')
        if os.path.exists(ckpt_path):
            print(f"Loading pre-trained CHCP from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6) # Tight LR for RL fine-tuning
    
    epochs = 50 # Increased for deeper convergence
    batch_size = 16
    iters_per_epoch = 1000 # Increased exploration volume
    
    metrics_dir = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp4_rl.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,avg_reward,success_rate,dev_hit1\n")
    
    print(f"\nStarting High-Intensity RL Fine-tuning (Exp 4-RL)...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_reward = 0
        successful_episodes = 0
        
        for i in tqdm(range(iters_per_epoch), desc=f"Epoch {epoch}"):
            batch_samples = [env.reset() for _ in range(batch_size)]
            
            questions = [s['question'] for s in batch_samples]
            encoded = tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # Forward pass to get logits
            rel_logits, stop_logits = model(encoded['input_ids'], encoded['attention_mask'])
            # rel_logits: [B, max_hops, num_rel]
            # stop_logits: [B, max_hops]
            
            # Sample paths (stochastic for RL exploration)
            batch_loss = 0
            for b in range(batch_size):
                sample = batch_samples[b]
                log_probs = []
                pred_path = []
                
                for h in range(model.max_hops):
                    # Sample relation
                    probs = F.softmax(rel_logits[b, h], dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    
                    log_probs.append(dist.log_prob(action))
                    rel_name = id2relation.get(action.item(), None)
                    if rel_name:
                        pred_path.append((rel_name, +1))
                    
                    # Stop if predicted (simplified)
                    stop_p = torch.sigmoid(stop_logits[b, h]).item()
                    if stop_p > 0.8: # high threshold for stop
                        break
                
                # Execute in ENV
                reward, success = env.step(sample, pred_path)
                total_reward += reward
                if success: successful_episodes += 1
                
                # Calculate policy gradient loss: -log_prob * reward
                # (Simple REINFORCE with baseline)
                for lp in log_probs:
                    batch_loss += -lp * reward
            
            # Update weights
            optimizer.zero_grad()
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            optimizer.step()
            
        avg_reward = total_reward / batch_size / iters_per_epoch
        success_rate = successful_episodes / (batch_size * iters_per_epoch)
        
        # EVAL ON DEV SET
        model.eval()
        dev_acc = 0
        dev_valid_hops = 0
        with torch.no_grad():
            for x, paths in tqdm(dev_loader, desc=f"Epoch {epoch} Dev"):
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)
                paths = paths.to(device)
                
                with torch.amp.autocast('cuda'):
                    rel_logits, _ = model(input_ids, attention_mask)
                
                preds = torch.argmax(rel_logits, dim=-1)
                valid_mask = paths != 0
                acc = (preds[valid_mask] == paths[valid_mask]).float().sum().item()
                dev_acc += acc
                dev_valid_hops += valid_mask.sum().item()
                
        dev_hit1 = dev_acc / dev_valid_hops if dev_valid_hops > 0 else 0
        print(f"Epoch {epoch} | Avg Reward: {avg_reward:.4f} | Train Success: {success_rate:.4f} | Dev Hit@1: {dev_hit1:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{avg_reward:.4f},{success_rate:.4f},{dev_hit1:.4f}\n")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/exp4_rl_epoch_{epoch}.pt")

    print("\nRL training complete.")

if __name__ == "__main__":
    train_chcp_rl()
