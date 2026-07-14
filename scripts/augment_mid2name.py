import json, os, sys, re
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

def augment_mid2name():
    print("Augmenting mid2name mapping from raw data...")
    
    mid2name_path = os.path.join(ROOT, 'data/master_mid2name.json')
    mid2name = {}
    if os.path.exists(mid2name_path):
        mid2name = json.load(open(mid2name_path, 'r', encoding='utf-8'))
    
    initial_count = len(mid2name)
    
    # Files to scan
    files = ['data/cwq_train.json', 'data/cwq_dev.json', 'data/cwq_test.json']
    
    for f_path in files:
        full_path = os.path.join(ROOT, f_path)
        if not os.path.exists(full_path): continue
        
        print(f"Scanning {f_path}...")
        data = json.load(open(full_path, 'r', encoding='utf-8'))
        
        for item in tqdm(data):
            # 1. Look for names in answers
            for ans in item.get('answers', []):
                mid = ans.get('answer_id', '').replace('ns:', '')
                name = ans.get('answer_name', '')
                if mid and name and mid not in mid2name:
                    mid2name[mid] = name
            
            # 2. Extract potential names from SPARQL comments or literals if any
            # (CWQ doesn't always have these, but worth a check)
            sparql = item.get('sparql', '')
            # Look for patterns like ns:m.0123 (Name)
            matches = re.findall(r"(ns:[mg]\.[\w\d_]+)\s+\(([^)]+)\)", sparql)
            for mid_raw, name in matches:
                mid = mid_raw.replace('ns:', '')
                if mid not in mid2name:
                    mid2name[mid] = name

    print(f"Augmentation complete. Added {len(mid2name) - initial_count} names.")
    print(f"Total names: {len(mid2name)}")
    
    with open(mid2name_path, 'w', encoding='utf-8') as f:
        json.dump(mid2name, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    augment_mid2name()
