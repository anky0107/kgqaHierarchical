import torch
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_collisions():
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    m2n_path = os.path.join(ROOT, 'data/master_mid2name.json')
    
    kg = torch.load(kg_path, map_location='cpu')
    forward = kg['forward']
    
    with open(m2n_path, 'r', encoding='utf-8') as f:
        m2n = json.load(f)
        
    names_to_check = ['San Francisco Giants', 'New York Yankees', 'Barack Obama']
    
    for name in names_to_check:
        mids = [m for m, n in m2n.items() if n == name]
        print(f"\n--- {name} ({len(mids)} MIDs) ---")
        for m in mids:
            degree = len(forward.get(m, []))
            print(f"  {m}: Degree {degree}")
            if degree > 0:
                print(f"    Sample: {forward[m][:2]}")

if __name__ == '__main__':
    check_collisions()
