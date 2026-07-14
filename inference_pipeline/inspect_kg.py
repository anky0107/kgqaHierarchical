import os, sys, torch, json
from collections import defaultdict

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def inspect_kg():
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    mid2name_path = os.path.join(ROOT, 'data/master_mid2name.json')

    if not os.path.exists(kg_path):
        print(f"Error: {kg_path} not found.")
        return

    print(f"Loading KG from {kg_path}...")
    kg = torch.load(kg_path, map_location='cpu')
    
    mid2name = {}
    if os.path.exists(mid2name_path):
        print(f"Loading entity names from {mid2name_path}...")
        mid2name = json.load(open(mid2name_path, 'r', encoding='utf-8'))

    def get_name(mid):
        return mid2name.get(mid, mid)

    # 1. Basic Stats
    forward = kg.get('forward', {})
    backward = kg.get('backward', {})
    
    all_entities = set(forward.keys()) | set(backward.keys())
    all_relations = set()
    num_triples = 0

    for ent, transitions in forward.items():
        num_triples += len(transitions)
        for rel, tgt in transitions:
            all_relations.add(rel)
            all_entities.add(tgt)

    print("\n" + "="*40)
    print("      KNOWLEDGE GRAPH STATISTICS")
    print("="*40)
    print(f"Total Unique Entities:  {len(all_entities):,}")
    print(f"Total Unique Relations: {len(all_relations):,}")
    print(f"Total Triples (Forward): {num_triples:,}")
    print("="*40)

    # 2. Display Sample Entities
    print("\n[Sample Entities and their Outgoing Relations]")
    sample_entities = list(forward.keys())[:5]
    for ent in sample_entities:
        name = get_name(ent)
        transitions = forward[ent]
        print(f"\nEntity: {name} ({ent})")
        for rel, tgt in transitions[:3]:
            print(f"  --[{rel}]--> {get_name(tgt)} ({tgt})")
        if len(transitions) > 3:
            print(f"  ... and {len(transitions)-3} more relations.")

    # 3. Display Top Relations
    rel_counts = defaultdict(int)
    for ent, transitions in forward.items():
        for rel, _ in transitions:
            rel_counts[rel] += 1
    
    sorted_rels = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n[Top 10 Most Frequent Relations]")
    for rel, count in sorted_rels[:10]:
        print(f"  {rel:50s} : {count:,} triples")

    # 4. Search Functionality (Interactive Example)
    print("\n" + "="*40)
    print("To search for a specific entity or MID, use:")
    print("  python inference_pipeline/inspect_kg.py <MID_or_Name>")
    print("="*40)

def search_entity(query):
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    mid2name_path = os.path.join(ROOT, 'data/master_mid2name.json')
    
    kg = torch.load(kg_path, map_location='cpu')
    mid2name = json.load(open(mid2name_path, 'r', encoding='utf-8'))
    name2mid = {v.lower(): k for k, v in mid2name.items()}

    # Try exact MID
    mid = None
    if query in mid2name:
        mid = query
    elif query.lower() in name2mid:
        mid = name2mid[query.lower()]
    else:
        # Partial name match
        for k, v in mid2name.items():
            if query.lower() in v.lower():
                mid = k
                print(f"Found partial match: {v} ({k})")
                break
    
    if not mid:
        print(f"Entity '{query}' not found.")
        return

    name = mid2name.get(mid, mid)
    print(f"\nResults for {name} ({mid}):")
    
    forward = kg.get('forward', {}).get(mid, [])
    backward = kg.get('backward', {}).get(mid, [])

    print(f"\nOutgoing Relations ({len(forward)}):")
    for rel, tgt in forward:
        print(f"  --[{rel}]--> {mid2name.get(tgt, tgt)} ({tgt})")

    print(f"\nIncoming Relations ({len(backward)}):")
    for rel, src in backward:
        print(f"  <--[{rel}]-- {mid2name.get(src, src)} ({src})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_entity(" ".join(sys.argv[1:]))
    else:
        inspect_kg()
