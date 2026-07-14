import json
import os
import random
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_selector_data(input_path, output_path, mid2name_path, num_negatives=5):
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loading mid2name from {mid2name_path}...")
    with open(mid2name_path, 'r', encoding='utf-8') as f:
        mid2name = json.load(f)
        
    all_mids = list(mid2name.keys())
    
    selector_data = []
    
    for item in data:
        question = item['question']
        gold_answers = item.get('answers', [])
        
        positives = []
        for ans in gold_answers:
            name = ans.get('answer')
            if name:
                positives.append(name)
        
        if not positives:
            continue
            
        # Add positives
        for pos in positives:
            selector_data.append({
                'question': question,
                'entity_name': pos,
                'label': 1
            })
            
        # Add negatives
        # In a real scenario, we'd pick hard negatives from the model's beam.
        # For now, we'll pick random ones to get the pipeline ready.
        negs_found = 0
        while negs_found < num_negatives * len(positives):
            neg_mid = random.choice(all_mids)
            neg_name = mid2name[neg_mid]
            if neg_name not in positives:
                selector_data.append({
                    'question': question,
                    'entity_name': neg_name,
                    'label': 0
                })
                negs_found += 1
                
    print(f"Generated {len(selector_data)} samples.")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selector_data, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    input_file = os.path.join(ROOT, 'data/cwq_train.json')
    output_file = os.path.join(ROOT, 'data/exp15_selector_train_data.json')
    mid2name_file = os.path.join(ROOT, 'data/master_mid2name.json')
    
    if os.path.exists(input_file) and os.path.exists(mid2name_file):
        generate_selector_data(input_file, output_file, mid2name_file)
    else:
        print("Required files missing.")
