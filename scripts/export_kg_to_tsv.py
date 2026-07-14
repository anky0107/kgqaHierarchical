import os, sys, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def export_kg_to_tsv():
    kg_path = os.path.join(ROOT, 'data/processed_kg/augmented_kg.pt')
    out_path = os.path.join(ROOT, 'data/processed_kg/readable_triples.tsv')

    if not os.path.exists(kg_path):
        print(f"Error: {kg_path} not found.")
        return

    print(f"Loading KG and exporting to {out_path}...")
    kg = torch.load(kg_path, map_location='cpu')
    forward = kg.get('forward', {})

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("Subject\tRelation\tObject\n")
        count = 0
        for subj, transitions in forward.items():
            for rel, obj in transitions:
                f.write(f"{subj}\t{rel}\t{obj}\n")
                count += 1
                if count % 10000 == 0:
                    print(f"  Exported {count} triples...")

    print(f"\nSuccess! You can now open {out_path} to see all details.")

if __name__ == '__main__':
    export_kg_to_tsv()
