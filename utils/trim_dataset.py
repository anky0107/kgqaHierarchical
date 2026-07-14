import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
in_file = os.path.join(ROOT, 'data/exp16v2_cds_train.json')
out_file = os.path.join(ROOT, 'data/exp16v2_cds_train_trimmed.json')

def process_chunk(chunk_str, is_first, is_last, out_f):
    if not chunk_str.strip(): return
    try:
        # Reconstruct valid json dict
        if not chunk_str.strip().startswith('{'): chunk_str = '{' + chunk_str
        if not chunk_str.strip().endswith('}'): chunk_str = chunk_str + '}'
        item = json.loads(chunk_str)
        
        golds = [c for c in item['candidates'] if c['is_gold']]
        negs = [c for c in item['candidates'] if not c['is_gold']]
        
        if golds:
            # Keep max 30 negatives
            item['candidates'] = golds + random.sample(negs, min(30, len(negs)))
            out_str = json.dumps(item, indent=2)
            if not is_first: out_f.write(',\n')
            out_f.write(out_str)
            return True
    except Exception as e:
        pass
    return False

print(f"Trimming 28GB dataset...")
with open(in_file, 'r', encoding='utf-8') as f_in, open(out_file, 'w', encoding='utf-8') as f_out:
    f_out.write('[\n')
    
    # Read line by line, accumulate a single dictionary
    current_item = []
    depth = 0
    first_written = True
    
    # Skip the first bracket
    f_in.readline()
    
    for line in f_in:
        line_stripped = line.strip()
        if line_stripped == '{' or line_stripped.startswith('{"'):
            depth += 1
        elif line_stripped == '},' or line_stripped == '}':
            depth -= 1
            
        current_item.append(line)
        
        if depth == 0 and len(current_item) > 1:
            chunk = "".join(current_item).strip()
            if chunk.endswith(','): chunk = chunk[:-1]
            success = process_chunk(chunk, first_written, False, f_out)
            if success: first_written = False
            current_item = []

    f_out.write('\n]')
print(f"Done! Saved trimmed dataset to {out_file}")
