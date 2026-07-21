import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Computes Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: list or tensor of step-level rewards
        values: list or tensor of state value predictions from critic
        gamma: discount factor
        lam: GAE lambda parameter controlling bias-variance tradeoff
        
    Returns:
        torch.Tensor of computed advantages
    """
    advantages = []
    gae = 0
    # Append 0 for terminal value to align lengths
    if isinstance(values, torch.Tensor):
        values = torch.cat([values, torch.tensor([0.0]).to(values.device)])
    else:
        values = values + [0.0]
        
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        
    return torch.tensor(advantages)
