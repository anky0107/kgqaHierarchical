import os, sys, torch, json
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

def audit_latest_kg():
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    mid2name_path = os.path.join(ROOT, 'data/master_mid2name.json')

    if not os.path.exists(kg_path):
        print(f"Error: {kg_path} not found. Run scripts/build_augmented_kg.py first.")
        return

    print(f"--- KG AUDIT REPORT: {kg_path} ---")
    kg = torch.load(kg_path, map_location='cpu')
    
    mid2name = {}
    if os.path.exists(mid2name_path):
        mid2name = json.load(open(mid2name_path, 'r', encoding='utf-8'))

    forward = kg.get('forward', {})
    backward = kg.get('backward', {})
    
    # 1. ENTITY ANALYSIS
    all_entities = set(forward.keys()) | set(backward.keys())
    virtual_entities = [e for e in all_entities if e.startswith('v:')]
    concrete_entities = [e for e in all_entities if not e.startswith('v:')]
    
    # 2. RELATION ANALYSIS
    rel_counts = defaultdict(int)
    num_triples = 0
    out_degrees = []
    
    for ent, transitions in forward.items():
        num_triples += len(transitions)
        out_degrees.append(len(transitions))
        for rel, tgt in transitions:
            rel_counts[rel] += 1
            all_entities.add(tgt) # Ensure targets are in the set

    # Re-calculate counts after target check
    virtual_entities = [e for e in all_entities if e.startswith('v:')]
    concrete_entities = [e for e in all_entities if not e.startswith('v:')]

    print("\n[1] ENTITY SUMMARY")
    print(f"  Total Unique Entities: {len(all_entities):,}")
    print(f"  Concrete MIDs:         {len(concrete_entities):,}")
    print(f"  Virtual Nodes (Vars):  {len(virtual_entities):,}")
    print(f"  Resolution Ratio:      {len(concrete_entities)/len(all_entities):.2%}")

    print("\n[2] GRAPH TOPOLOGY")
    print(f"  Total Forward Triples: {num_triples:,}")
    print(f"  Unique Relations:      {len(rel_counts):,}")
    if out_degrees:
        print(f"  Avg Out-Degree:        {np.mean(out_degrees):.2f}")
        print(f"  Max Out-Degree:        {np.max(out_degrees):,}")
    
    print("\n[3] TOP 20 RELATIONS")
    sorted_rels = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (rel, count) in enumerate(sorted_rels[:20]):
        print(f"  {i+1:2d}. {rel:50s} : {count:,}")

    print("\n[4] VIRTUAL NODE SAMPLE")
    for v_ent in virtual_entities[:5]:
        print(f"  Node: {v_ent}")
        neighbors = forward.get(v_ent, [])
        for rel, tgt in neighbors[:3]:
            tgt_name = mid2name.get(tgt, tgt)
            print(f"    --[{rel}]--> {tgt_name} ({tgt})")
        if len(neighbors) > 3:
            print(f"    ... + {len(neighbors)-3} more")

    print("\n[5] CONCRETE NODE SAMPLE (High Degree)")
    high_degree_nodes = sorted(forward.keys(), key=lambda x: len(forward[x]), reverse=True)
    concrete_high = [n for n in high_degree_nodes if not n.startswith('v:')][:5]
    for c_ent in concrete_high:
        name = mid2name.get(c_ent, c_ent)
        print(f"  Node: {name} ({c_ent}) - Degree: {len(forward[c_ent])}")
        for rel, tgt in forward[c_ent][:3]:
            print(f"    --[{rel}]--> {mid2name.get(tgt, tgt)}")

    print("\n--- END OF REPORT ---")

if __name__ == "__main__":
    audit_latest_kg()
