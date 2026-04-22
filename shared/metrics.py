import torch

def hits_at_k(preds, targets, k=1):
    """
    preds: tensor of shape [batch_size, num_classes] (logits or probs)
    targets: tensor of shape [batch_size]
    """
    topk = torch.topk(preds, k=k, dim=-1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float().sum()
    return correct.item()

def mean_reciprocal_rank(preds, targets):
    batch_size = preds.size(0)
    sorted_indices = torch.argsort(preds, dim=-1, descending=True)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    mrr = (1.0 / ranks.float()).sum()
    return mrr.item()
