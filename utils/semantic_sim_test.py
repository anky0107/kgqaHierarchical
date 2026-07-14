"""
Quick diagnostic: Does RoBERTa understand semantic relation similarity?
Tests whether raw RoBERTa (mean-pool) can rank the correct relation
higher for a given question, WITHOUT any fine-tuning.
"""
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
model.eval()

def encode_mean(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    return (out.last_hidden_state * mask).sum(1) / mask.sum(1)

def rel_to_text(rel):
    """Convert Freebase relation to natural language."""
    return rel.replace('.', ' ').replace('_', ' ')

# --- TEST 1: Concert Tour Question ---
print("=" * 60)
print("TEST 1: Concert tour bridge entity question")
print("=" * 60)
question1 = "Where did the Country Nation World Tour concert artist go to college"
q1_emb = encode_mean(question1)

relations_test1 = [
    ('music.concert_tour.artist',        'CORRECT hop1 - tour to artist'),
    ('music.artist.concert_tours',       'WRONG direction hop1'),
    ('education.education.institution',  'CORRECT hop2 - artist to college'),
    ('people.person.education',          'alt hop2 - person schooling'),
    ('music.artist.genre',               'music but irrelevant'),
    ('film.actor.film',                  'wrong domain'),
    ('government.politician.party',      'wrong domain'),
]

sims = []
for rel, label in relations_test1:
    emb = encode_mean(rel_to_text(rel))
    sim = F.cosine_similarity(q1_emb, emb).item()
    sims.append((sim, rel, label))

sims.sort(reverse=True)
print(f"Question: \"{question1}\"\n")
print("Ranked by cosine similarity (mean-pool):")
for sim, rel, label in sims:
    marker = " <-- WANT THIS" if "CORRECT" in label else ""
    print(f"  {sim:.4f}  {rel}  [{label}]{marker}")

# --- TEST 2: Simple 1-hop question (sanity check) ---
print()
print("=" * 60)
print("TEST 2: Sanity check - simple 1-hop question")
print("=" * 60)
question2 = "What genre of music does Taylor Swift play"
q2_emb = encode_mean(question2)

relations_test2 = [
    ('music.artist.genre',           'CORRECT'),
    ('music.artist.album',           'music but wrong'),
    ('education.education.institution', 'wrong domain'),
    ('film.actor.film',              'wrong domain'),
    ('people.person.nationality',    'wrong domain'),
]

sims2 = []
for rel, label in relations_test2:
    emb = encode_mean(rel_to_text(rel))
    sim = F.cosine_similarity(q2_emb, emb).item()
    sims2.append((sim, rel, label))

sims2.sort(reverse=True)
print(f"Question: \"{question2}\"\n")
print("Ranked by cosine similarity (mean-pool):")
for sim, rel, label in sims2:
    marker = " <-- WANT THIS" if "CORRECT" in label else ""
    print(f"  {sim:.4f}  {rel}  [{label}]{marker}")

print()
print("CONCLUSION:")
print("If TEST 2 ranks 'music.artist.genre' correctly but TEST 1 fails,")
print("it means RoBERTa understands direct question-relation mapping")
print("but lacks multi-hop semantic bridging. This confirms the need for")
print("CONTRASTIVE training over relation TEXTS, not just gold path IDs.")
