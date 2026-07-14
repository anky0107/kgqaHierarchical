import json
import os
import sys
import torch
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ingest_kb_cache():
    cache_path = os.path.join(ROOT, 'data/cwq_ianyunshi/CWQ/kb_cache.json')
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found at {cache_path}")
        return

    print(f"Loading existing KG from {kg_path}...")
    kg_data = torch.load(kg_path)
    
    # Existing structure is dict with 'forward' and 'backward'
    # forward[s] = [(rel, obj), ...]
    # backward[o] = [(rel, subj), ...]
    
    forward = defaultdict(set)
    backward = defaultdict(set)
    
    # Load existing into sets for fast merging
    for s, transitions in kg_data.get('forward', {}).items():
        for rel, obj in transitions:
            forward[s].add((rel, obj))
            
    for o, transitions in kg_data.get('backward', {}).items():
        for rel, subj in transitions:
            backward[o].add((rel, subj))

    print(f"Initial Subject Nodes: {len(forward)}")

    print(f"Loading cache from {cache_path}...")
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    print(f"Processing {len(cache_data)} entities from cache...")
    
    new_triples_count = 0
    
    for mid, queries in tqdm(cache_data.items()):
        s = mid.replace('ns:', '')
        
        for query_str, results in queries.items():
            parts = query_str.split()
            
            # 1-hop: "m.094w0s people.person.gender ?e1"
            if len(parts) == 3 and parts[2] == '?e1':
                r = parts[1].replace('ns:', '')
                for res in results:
                    o = str(res).replace('ns:', '')
                    if (r, o) not in forward[s]:
                        forward[s].add((r, o))
                        backward[o].add((r, s))
                        new_triples_count += 1
            
            # 2-hop: "m.094w0s people.person.education ?d1\t?d1 education.education.degree ?e2"
            elif len(parts) == 6 and parts[2] == '?d1' and parts[3] == '\t?d1' and parts[5] == '?e2':
                r1 = parts[1].replace('ns:', '')
                r2 = parts[4].replace('ns:', '')
                r_comp = f"{r1}..{r2}"
                for res in results:
                    o = str(res).replace('ns:', '')
                    if (r_comp, o) not in forward[s]:
                        forward[s].add((r_comp, o))
                        backward[o].add((r_comp, s))
                        new_triples_count += 1

    print(f"Done. Added {new_triples_count} new triples from cache.")
    
    # Convert sets back to lists
    final_forward = {s: list(trans) for s, trans in forward.items()}
    final_backward = {o: list(trans) for o, trans in backward.items()}

    print(f"Final Subject Nodes: {len(final_forward)}")
    print(f"Final Object Nodes: {len(final_backward)}")

    print(f"Saving expanded KG to {kg_path}...")
    torch.save({'forward': final_forward, 'backward': final_backward}, kg_path)
    
    # Export TSV (limited to first 100k for sanity, or all if you want)
    tsv_path = os.path.join(ROOT, 'data/processed_kg/readable_triples.tsv')
    print(f"Exporting sample to {tsv_path}...")
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("Subject\tRelation\tObject\n")
        count = 0
        for s, transitions in final_forward.items():
            for r, o in transitions:
                f.write(f"{s}\t{r}\t{o}\n")
                count += 1
                if count >= 1000000: break # Cap at 1M triples for readable file
            if count >= 1000000: break
            
    print("Success.")

if __name__ == '__main__':
    ingest_kb_cache()
