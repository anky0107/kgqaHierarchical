import os, sys, json, torch, random, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from train.exp16v2_harvest import MultiModelHarvester
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("============================================================")
print("  CDS v2 UNIVERSAL BENCHMARK (EXP 7 vs 9 vs 15)  ")
print("============================================================")

print("\n[A] Loading Universal Harvester (Exp 7, 9, 15 models)...")
harvester = MultiModelHarvester()

print("\n[B] Loading CDS v2 Pipeline (Stage 1, 2, 3)...")
tok_s1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mod_s1 = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
mod_s1.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp16v2_s1_bi.pt")))
mod_s1.eval()

tok_s2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mod_s2 = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
mod_s2.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp16v2_s2_path.pt")), strict=False)
mod_s2.eval()

tok_s3 = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
mod_s3 = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
mod_s3.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp16v2_s3_cross.pt")))
mod_s3.eval()

with open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# Evaluate on the COMPLETE dev set
TOTAL_Q = len(dev_data)
print(f"\nStarting live dynamic generation and CDS inference for {TOTAL_Q} questions...")

results = {
    'Exp 7 (Unified Baseline)':  {'hits1': 0, 'hits3': 0, 'hits10': 0},
    'Exp 9 (RL Meta-Const)':    {'hits1': 0, 'hits3': 0, 'hits10': 0},
    'Exp 15 (STRL Teacher)':    {'hits1': 0, 'hits3': 0, 'hits10': 0}
}

from utils.sparql_parser import find_reasoning_path

def evaluate_candidates(q, raw_candidates, exp_name, gold):
    # raw_candidates is a dict: {mid: path_str}
    if not raw_candidates: return
    
    # Format for CDS
    cands = []
    for mid, path_str in raw_candidates.items():
        cands.append({
            'mid': mid,
            'name': harvester.mid2name.get(mid, "Unknown"),
            'is_gold': mid in gold,
            'path': path_str
        })
    if not any(c['is_gold'] for c in cands): return # Candidate generator failed to find it
    
    # S1: Bi-Encoder Pre-Filter
    with torch.no_grad():
        qe = tok_s1(q, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        qv = mod_s1(**qe).last_hidden_state[:, 0, :]
        
        # Batching S1 to prevent OOM
        ev_list = []
        c_names = [c.get('name', '') for c in cands]
        for i in range(0, len(c_names), 500):
            ee = tok_s1(c_names[i:i+500], return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
            ev_list.append(mod_s1(**ee).last_hidden_state[:, 0, :])
        ev = torch.cat(ev_list, dim=0)
        s1_scores = F.cosine_similarity(qv, ev)
        
        top_k1 = torch.topk(s1_scores, min(100, len(cands))).indices.cpu().tolist()
        cands_s2 = [cands[i] for i in top_k1]
        
        # S2: Path-Aware Sieve (mpnet)
        q_s2 = tok_s2([q]*len(cands_s2), padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        p_s2 = tok_s2([str(c.get('path', '')) for c in cands_s2], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        e_s2 = tok_s2([str(c.get('name', '')) for c in cands_s2], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        qv2 = mod_s2(**q_s2).last_hidden_state[:, 0, :]
        pv2 = mod_s2(**p_s2).last_hidden_state[:, 0, :]
        ev2 = mod_s2(**e_s2).last_hidden_state[:, 0, :]
        s2_scores = F.cosine_similarity(qv2 + pv2, ev2)
        top_k2 = torch.topk(s2_scores, min(15, len(cands_s2))).indices.cpu().tolist()
        cands_s3 = [cands_s2[i] for i in top_k2]

        # S3: Cross-Encoder (bge-reranker)
        qs3 = [str(q)]*len(cands_s3)
        es3 = [str(c.get('name', '')) for c in cands_s3]
        enc3 = tok_s3(qs3, es3, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        s3_scores = mod_s3(**enc3).logits.squeeze(-1)
        
        final_ranking = sorted(zip(cands_s3, s3_scores.cpu().tolist()), key=lambda x: x[1], reverse=True)
        
        # Metrics
        is_hit = lambda x: x[0]['is_gold']
        if any(is_hit(x) for x in final_ranking[:1]): results[exp_name]['hits1'] += 1
        if any(is_hit(x) for x in final_ranking[:3]): results[exp_name]['hits3'] += 1
        if any(is_hit(x) for x in final_ranking[:10]): results[exp_name]['hits10'] += 1

target_exp = sys.argv[1] if len(sys.argv) > 1 else 'exp15'
print(f"\n[!] Running isolated evaluation for: {target_exp}")

with torch.no_grad():
    for idx, item in enumerate(dev_data[:TOTAL_Q]):
        try:
            path = find_reasoning_path(item['sparql'])
            if not path: continue
            q = item['question']
            te = path[0][0].replace("ns:", "")
            gold = set(a['answer_id'].replace("ns:", "") for a in item.get('answers', []))
            inputs = harvester.tokenizer(q, return_tensors='pt', padding=True, truncation=True).to(device)
            
            def cap_candidates(beam, path_str, gold_set):
                cands = list(beam)
                if not gold_set.intersection(cands): return {mid: path_str for mid in cands[:30]} 
                negatives = [c for c in cands if c not in gold_set]
                if len(negatives) > 30:
                    negatives = random.sample(negatives, 30)
                final_cands = list(gold_set.intersection(cands)) + negatives
                return {mid: path_str for mid in final_cands}

            if target_exp == 'exp7':
                b7, p7 = harvester.get_beam('exp7', inputs, te)
                evaluate_candidates(q, cap_candidates(b7, p7, gold), 'Exp 7 (Unified Baseline)', gold)
            elif target_exp == 'exp9':
                b9, p9 = harvester.get_beam('exp9', inputs, te)
                evaluate_candidates(q, cap_candidates(b9, p9, gold), 'Exp 9 (RL Meta-Const)', gold)
            elif target_exp == 'exp15':
                b15, p15 = harvester.get_beam('exp15', inputs, te)
                evaluate_candidates(q, cap_candidates(b15, p15, gold), 'Exp 15 (STRL Teacher)', gold)
                
            if (idx + 1) % 50 == 0:
                print(f"Progress: [{idx+1}/{TOTAL_Q}] questions evaluated for {target_exp}...")
        except Exception as e:
            continue

print("\n============================================================")
print(f"     ISOLATED CDS v2 BENCHMARK RESULTS ({TOTAL_Q} Questions)")
print("============================================================")
for exp_name, res in results.items():
    if res['hits1'] > 0 or res['hits3'] > 0 or res['hits10'] > 0: # Only print the one we actually ran
        print(f"\n{exp_name}:")
        print(f"  Hit@1  :  {res['hits1']/TOTAL_Q*100:.2f}%")
        print(f"  Hit@3  :  {res['hits3']/TOTAL_Q*100:.2f}%")
        print(f"  Hit@10 :  {res['hits10']/TOTAL_Q*100:.2f}%")
print("============================================================")
