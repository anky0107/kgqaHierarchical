import os, sys
# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from shared.kg_loader import build_kg_from_cwq_triples
from utils.sparql_parser import extract_triples


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
proc_dir = os.path.join(ROOT, 'data', 'processed_universal')
# Paths to CWQ JSON files (train, dev, test)
cwq_files = [
    os.path.join(ROOT, 'data', 'cwq_train.json'),
    os.path.join(ROOT, 'data', 'cwq_dev.json'),
    os.path.join(ROOT, 'data', 'cwq_test.json')
]

print('Building Knowledge Graph from CWQ SPARQL files...')
kg = build_kg_from_cwq_triples(cwq_files, extractor_fn=extract_triples)

kg_path = os.path.join(proc_dir, 'kg.pt')
print(f'Saving KG to {kg_path}')
torch.save(kg, kg_path)
print('Done.')
