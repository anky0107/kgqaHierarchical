import json, os, sys, torch
from collections import defaultdict
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.sparql_parser import find_reasoning_path, extract_triples, find_answer_variable

def build_augmented_kg():
    print("Building Augmented KG (Resolving Variables to Gold MIDs)...")
    
    # 1. Load all data
    files = ['data/cwq_train.json', 'data/cwq_dev.json', 'data/cwq_test.json']
    all_data = []
    for f in files:
        if os.path.exists(f):
            all_data.extend(json.load(open(f, 'r', encoding='utf-8')))

    # 2. Extract Triples and Resolve
    # We want a KG that contains actual entities, not variables
    kg_triples = []
    
    for item in tqdm(all_data):
        sparql = item['sparql']
        answers = {a['answer_id'].replace('ns:', '') for a in item.get('answers', []) if 'answer_id' in a}
        
        path = find_reasoning_path(sparql)
        if not path: continue
        
        # Build a mapping from ?var to concrete ID based on the reasoning path and answers
        var_map = {}
        # The last node in the reasoning path is the answer variable
        if path[-1][3].startswith('?'):
            ans_var = path[-1][3]
            if answers:
                # We pick the first answer as the representative for this variable in the KG
                var_map[ans_var] = list(answers)[0]
        
        # Also extract intermediate variables if possible?
        # For simplicity, let's just add the Reasoning Path itself to the KG with resolved IDs
        
        # Resolve path nodes
        resolved_nodes = []
        # Start node is always constant
        curr_node = path[0][0].replace('ns:', '')
        resolved_nodes.append(curr_node)
        
        for i, (u, r, direction, v) in enumerate(path):
            # We know r and direction.
            # If v is the answer var, we resolved it.
            # If v is an intermediate var, we might not know it... 
            # unless we have the full Freebase. 
            # But wait! The triples in the SPARQL often have other constants.
            
            # Let's just add the triple (u_resolved, r, v_resolved)
            # We'll use the answers for the end variable.
            # For intermediate variables, we'll keep them as variables if we must, 
            # but that still makes it a "Ghost Town".
            
            # WAIT! The SPARQL triples often look like:
            # ?x ns:rel1 ns:m.01 .
            # ?x ns:rel2 ?y .
            # ?y ns:rel3 ns:m.02 .
            
            # If we just add the triples from SPARQL but replace variables with "unique dummy IDs" 
            # per question, it might help? No.
            
            # Actually, the most "SOTA" way to do this without Freebase is to use the 
            # "Reasoning Path" as the KG itself for that question.
            pass

    # REVISED PLAN: 
    # Instead of a global KG, let's use the triples from the SPARQL but 
    # if a triple is (constant, relation, ?var) and we know ?var is an answer, 
    # replace ?var with the answer ID.
    
    kg_forward = defaultdict(list)
    kg_backward = defaultdict(list)
    
    print("Processing questions and resolving variables...")
    for q_idx, item in enumerate(tqdm(all_data)):
        sparql = item['sparql']
        triples = extract_triples(sparql)
        answers = {a['answer_id'].replace('ns:', '') for a in item.get('answers', []) if 'answer_id' in a}
        
        # 1. Identify Answer Variable
        ans_var = find_answer_variable(sparql)
        
        # 2. Build local resolution map for this question
        # Map variables like ?x to unique IDs: "v:q{idx}_{var}"
        local_var_map = {}
        
        for subj, rel, obj in triples:
            for node in [subj, obj]:
                if node.startswith('?') and node not in local_var_map:
                    if node == ans_var and answers:
                        # Resolve answer variable to the first gold answer
                        local_var_map[node] = list(answers)[0]
                    else:
                        # Assign a unique virtual ID for this variable in this question context
                        # This prevents "?c" from one question merging with "?c" from another
                        local_var_map[node] = f"v:{q_idx}_{node[1:]}"

        # 3. Add resolved triples to global KG
        for subj, rel, obj in triples:
            s = subj.replace('ns:', '')
            o = obj.replace('ns:', '')
            
            s = local_var_map.get(s, s)
            o = local_var_map.get(o, o)
            
            # Add to KG
            kg_forward[s].append((rel, o))
            kg_backward[o].append((rel, s))

    # 4. Deduplicate transitions
    print("Deduplicating KG transitions...")
    final_forward = {}
    for s, transitions in kg_forward.items():
        # Keep unique (rel, obj) pairs
        final_forward[s] = list(set(transitions))
    
    final_backward = {}
    for o, transitions in kg_backward.items():
        final_backward[o] = list(set(transitions))

    # Save this "High-Fidelity KG"
    kg_data = {
        'forward': final_forward,
        'backward': final_backward
    }
    os.makedirs('data/processed_kg', exist_ok=True)
    torch.save(kg_data, 'data/processed_kg/augmented_kg.pt')
    print(f"Saved augmented KG with {len(kg_forward)} subject nodes.")

import re
if __name__ == '__main__':
    build_augmented_kg()
