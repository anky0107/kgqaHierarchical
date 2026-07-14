"""
Test: Can RoBERTa provide per-hop semantic teacher signal?
This is the core question for the Semantic-Teacher RL architecture.
"""
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
model.eval()

def encode_mean(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    return (out.last_hidden_state * mask).sum(1) / mask.sum(1)

rel2id = torch.load('data/processed_entity/relation2id.pt')
all_rels = list(rel2id.keys())

print('Pre-encoding all 645 relations...')
rel_embs = {}
for rel in all_rels:
    text = rel.replace('.', ' ').replace('_', ' ')
    rel_embs[rel] = encode_mean(text)
print('Done.')

def top_k_semantic(hop_q, k=5):
    q_emb = encode_mean(hop_q)
    scores = [(F.cosine_similarity(q_emb, rel_embs[r]).item(), r) for r in all_rels]
    scores.sort(reverse=True)
    return scores[:k]

# The key: teacher uses hop-decomposed queries
# Hop 1 context: what is the BRIDGE entity we need to find first?
# Hop 2 context: what is the FINAL answer entity type?

print()
print("HOP 1 - Concert tour bridge:")
for sim, rel in top_k_semantic('concert tour artist performer musician'):
    print(f'  {sim:.4f}  {rel}')

print()
print("HOP 2 - College/education:")
for sim, rel in top_k_semantic('college university school attended education'):
    print(f'  {sim:.4f}  {rel}')

print()
print("HOP 1 - Simple genre question (sanity):")
for sim, rel in top_k_semantic('music genre style artist'):
    print(f'  {sim:.4f}  {rel}')

print()
print("HOP 1 - Film director bridge:")
for sim, rel in top_k_semantic('film movie directed by director'):
    print(f'  {sim:.4f}  {rel}')
