import torch
import os
import re

def tsv_to_pt(tsv_path, output_path):
    print(f"Converting {tsv_path} to {output_path}...")
    forward = {}
    backward = {}
    
    # Regex to extract MIDs from <http://rdf.freebase.com/ns/m.01234>
    mid_regex = re.compile(r'<http://rdf.freebase.com/ns/([mg]\.[a-z0-9_]+)>')
    
    count = 0
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 1000000 == 0:
                print(f"Processed {count//1000000}M triples...")
                
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
                
            subj_raw = parts[0]
            pred_raw = parts[1]
            obj_raw = parts[2]
            
            # Extract Subject
            m_s = mid_regex.search(subj_raw)
            if not m_s: continue
            subj = m_s.group(1)
            
            # Extract Predicate (Relation)
            # Relations are often ns:film.film.directed_by
            m_p = mid_regex.search(pred_raw)
            if m_p:
                pred = m_p.group(1)
            else:
                # Handle standard RDF relations or property names
                pred = pred_raw.split('/')[-1].replace('>', '')
                
            # Extract Object
            # Could be a MID or a Literal
            m_o = mid_regex.search(obj_raw)
            if m_o:
                obj = m_o.group(1)
                # Save Triple
                if subj not in forward: forward[subj] = []
                forward[subj].append((pred, obj))
                
                if obj not in backward: backward[obj] = []
                backward[obj].append((pred, subj))
            else:
                # It's a literal (e.g., "Inception"@en)
                # We can store literals in a separate dict if needed
                pass

    print(f"Saving to {output_path}...")
    torch.save({'forward': forward, 'backward': backward}, output_path)
    print("Done!")

if __name__ == "__main__":
    TSV_PATH = 'data/cwq_filtered_kg.tsv'
    OUTPUT_PATH = 'data/processed_kg/augmented_kg.pt'
    
    if os.path.exists(TSV_PATH):
        tsv_to_pt(TSV_PATH, OUTPUT_PATH)
    else:
        print(f"Error: {TSV_PATH} not found.")
