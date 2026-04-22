# scripts/generate_rel_embeddings.py
import torch
from transformers import RobertaModel, RobertaTokenizer
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
rel2id_path = os.path.join(ROOT, 'data', 'processed_universal', 'relation2id.pt')
emb_path = os.path.join(ROOT, 'data', 'processed_universal', 'relation_embeddings.pt')

print('Loading relation2id...')
relation2id = torch.load(rel2id_path)
relations = [rel for rel, _ in sorted(relation2id.items(), key=lambda x: x[1])]

print('Loading RoBERTa tokenizer and model...')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
model.eval()

embeddings = []
for rel in relations:
    inputs = tokenizer(rel, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)
        embeddings.append(cls_vec)

emb_tensor = torch.stack(embeddings)
print(f'Saving {emb_tensor.shape[0]} relation embeddings to {emb_path}')
torch.save(emb_tensor, emb_path)
print('Done.')
