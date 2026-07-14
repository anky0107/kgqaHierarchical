import json
import re
import os

def extract_mids_from_sparql(sparql):
    # Matches ns:m.XXXX or ns:g.XXXX
    return re.findall(r'ns:([mg]\.[a-z0-9_]+)', sparql)

def get_all_cwq_mids(data_dir):
    mids = set()
    files = ['cwq_train.json', 'cwq_dev.json', 'cwq_test.json']
    
    for filename in files:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        print(f"Processing {filename}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # 1. From answers
                for ans in item.get('answers', []):
                    if 'answer_id' in ans:
                        mids.add(ans['answer_id'].replace('.', '_')) # Freebase RDF uses underscores often
                        mids.add(ans['answer_id']) # Keep both versions for safety
                
                # 2. From SPARQL
                if 'sparql' in item:
                    found = extract_mids_from_sparql(item['sparql'])
                    for f_mid in found:
                        mids.add(f_mid)
                        mids.add(f_mid.replace('.', '_'))

    # Also clean the MIDs to match the RDF format (usually m.XXXX)
    cleaned_mids = set()
    for mid in mids:
        # Standardize to m.XXXX or g.XXXX
        m = re.match(r'([mg])[\._]([a-z0-9_]+)', mid)
        if m:
            cleaned_mids.add(f"{m.group(1)}.{m.group(2)}")
            
    return cleaned_mids

if __name__ == "__main__":
    DATA_DIR = 'data'
    all_mids = get_all_cwq_mids(DATA_DIR)
    print(f"Extracted {len(all_mids)} unique MIDs.")
    
    with open('data/cwq_mids.txt', 'w') as f:
        for mid in sorted(all_mids):
            f.write(mid + '\n')
    print("Saved to data/cwq_mids.txt")
