import os, json, torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("============================================================")
print("  CDS v2 (EXP 16v2) FINAL BENCHMARK  ")
print("============================================================")

print("[1/3] Loading Stage 1 (Bi-Encoder: all-MiniLM-L6-v2) ...")
tok_s1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mod_s1 = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
mod_s1.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp16v2_s1_bi.pt")))
mod_s1.eval()

print("[2/3] Loading Stage 2 (Path-Sieve: all-mpnet-base-v2) ...")
tok_s2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mod_s2 = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
mod_s2.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp16v2_s2_path.pt")), strict=False)
mod_s2.eval()

print("[3/3] Loading Stage 3 (Cross-Encoder: bge-reranker-base) ...")
tok_s3 = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
mod_s3 = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
mod_s3.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/exp16v2_s3_cross.pt")))
mod_s3.eval()

dev_file = os.path.join(ROOT, "data/exp16_cds_dev.json")
print(f"\nLoading dev dataset: {dev_file}")
with open(dev_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fast inference loop
hits_1, hits_3, hits_10 = 0, 0, 0
total = min(len(data), 500)  # Evaluate on first 500 for fast thesis benchmark

print(f"\nRunning 3-Stage Cascading Evaluation on {total} questions...")
with torch.no_grad():
    for item in tqdm(data[:total], desc="Evaluating"):
        q = item['question']
        cands = item['candidates']
        
        # S1: Bi-Encoder Pre-Filter
        qe = tok_s1(q, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        qv = mod_s1(**qe).last_hidden_state[:, 0, :]
        
        # Prevent OOM by chunking the candidate evaluation (some questions have 6000+ cands)
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
        p_s2 = tok_s2([c.get('path', '') for c in cands_s2], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        e_s2 = tok_s2([c['name'] for c in cands_s2], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        qv2 = mod_s2(**q_s2).last_hidden_state[:, 0, :]
        pv2 = mod_s2(**p_s2).last_hidden_state[:, 0, :]
        ev2 = mod_s2(**e_s2).last_hidden_state[:, 0, :]
        s2_scores = F.cosine_similarity(qv2 + pv2, ev2)
        top_k2 = torch.topk(s2_scores, min(15, len(cands_s2))).indices.cpu().tolist()
        cands_s3 = [cands_s2[i] for i in top_k2]

        # S3: Cross-Encoder (bge-reranker)
        qs3 = [q]*len(cands_s3)
        es3 = [c['name'] for c in cands_s3]
        enc3 = tok_s3(qs3, es3, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        s3_scores = mod_s3(**enc3).logits.squeeze(-1)
        
        final_ranking = sorted(zip(cands_s3, s3_scores.cpu().tolist()), key=lambda x: x[1], reverse=True)
        
        # Metrics
        is_hit = lambda x: x[0]['is_gold']
        if any(is_hit(x) for x in final_ranking[:1]): hits_1 += 1
        if any(is_hit(x) for x in final_ranking[:3]): hits_3 += 1
        if any(is_hit(x) for x in final_ranking[:10]): hits_10 += 1

print("\n============================================================")
print("                 FINAL CDS v2 BENCHMARK RESULTS")
print("============================================================")
print(f" Hit@1  :  {hits_1/total*100:.2f}%")
print(f" Hit@3  :  {hits_3/total*100:.2f}%")
print(f" Hit@10 :  {hits_10/total*100:.2f}%")
print("============================================================")
if hits_1/total > 0.40:
    print("SUCCESS: 40% accuracy threshold crossed! Thesis benchmark complete.")
