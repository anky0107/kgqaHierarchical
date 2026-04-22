import torch

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().sum()
    return correct / len(targets)

def topk_accuracy(logits, targets, k=3):
    topk = torch.topk(logits, k=k, dim=-1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float().sum()
    return correct / len(targets)