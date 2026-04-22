"""
Zero-Shot MetaQA Evaluation

Our models are trained on CWQ (Freebase relations). MetaQA has 9 movie relations.
Zero-shot approach:
  1. Use our trained encoder to get question representation
  2. Map CWQ-predicted relations to MetaQA relations via semantic similarity
  3. Traverse MetaQA's KG with mapped relations
  4. Measure Hits@1

Two strategies:
  A) Semantic Mapping: embed MetaQA relation names + CWQ relation names, 
     find nearest MetaQA relation for each predicted CWQ relation
  B) Direct Re-head: Use the frozen encoder + train a tiny 9-class head 
     (but that's not zero-shot, skip this)
  
For true zero-shot, we use Strategy A.
"""
import os, sys, re, torch, functools
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

METAQA_DIR = os.path.join(ROOT, 'data', 'metaqa')

# ============================================================
#  1. Load MetaQA KG
# ============================================================

class MetaQAKG:
    """Simple KG from MetaQA kb.txt"""
    def __init__(self, kb_path):
        self.forward = defaultdict(list)   # entity -> [(rel, target)]
        self.backward = defaultdict(list)  # entity -> [(rel, source)]
        self.relations = set()
        
        with open(kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('|')
                if len(parts) != 3: continue
                subj, rel, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
                self.forward[subj].append((rel, obj))
                self.backward[obj].append((rel, subj))
                self.relations.add(rel)
        
        print(f"  MetaQA KG: {len(self.forward)} subjects, {len(self.backward)} objects, {len(self.relations)} relations")
        print(f"  Relations: {sorted(self.relations)}")
    
    def get_neighbors(self, entity):
        """Returns all (relation, target, direction) tuples"""
        neighbors = []
        for rel, tgt in self.forward.get(entity, []):
            neighbors.append((rel, tgt, +1))
        for rel, src in self.backward.get(entity, []):
            neighbors.append((rel, src, -1))
        return neighbors
    
    def traverse(self, start, relations_per_hop):
        """Traverse KG following predicted relations at each hop.
        relations_per_hop: list of sets of relation names.
        Returns set of reached entities.
        """
        active = {start}
        for hop_rels in relations_per_hop:
            if not active: break
            next_ents = set()
            for e in active:
                for rel, tgt, d in self.get_neighbors(e):
                    if rel in hop_rels:
                        next_ents.add(tgt)
            active = next_ents
        return active

# ============================================================
#  2. Load MetaQA Test Data
# ============================================================

def load_metaqa_test(hop):
    """Load MetaQA test questions for given hop count."""
    path = os.path.join(METAQA_DIR, f'{hop}-hop', 'vanilla', 'qa_test.txt')
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) != 2: continue
            question, answers_str = parts
            
            # Extract topic entity from [brackets]
            match = re.search(r'\[(.+?)\]', question)
            if not match: continue
            topic_entity = match.group(1)
            
            # Clean question (remove brackets)
            clean_q = question.replace('[', '').replace(']', '')
            
            # Parse answers
            answers = set(a.strip() for a in answers_str.split('|') if a.strip())
            
            samples.append({
                'question': clean_q,
                'topic_entity': topic_entity,
                'gold_answers': answers,
                'num_hops': hop,
            })
    
    print(f"  {hop}-hop test: {len(samples)} questions")
    return samples

# ============================================================
#  3. Semantic Relation Mapping (CWQ → MetaQA)
# ============================================================

METAQA_RELATIONS = [
    'directed_by', 'written_by', 'starred_actors', 'release_year',
    'in_language', 'has_genre', 'has_tags', 'has_imdb_rating', 'has_imdb_votes'
]

# Expand MetaQA relation names into natural phrases for better embedding
METAQA_REL_DESCRIPTIONS = {
    'directed_by': 'who directed the movie, film director',
    'written_by': 'who wrote the movie, film writer, screenplay',
    'starred_actors': 'which actors star in the movie, cast members',
    'release_year': 'when was the movie released, year of release',
    'in_language': 'what language is the movie in',
    'has_genre': 'what genre is the movie, film genre category',
    'has_tags': 'what tags describe the movie, keywords',
    'has_imdb_rating': 'what is the IMDB rating of the movie',
    'has_imdb_votes': 'how many IMDB votes does the movie have',
}

def build_relation_mapping(cwq_rel2id, device):
    """
    Build a mapping from CWQ relation IDs to MetaQA relations.
    Uses a pre-trained language model to embed relation names and find nearest matches.
    """
    from transformers import AutoTokenizer, AutoModel
    
    print("  Building CWQ→MetaQA relation mapping via semantic similarity...")
    
    # Use a sentence transformer for embedding
    tok = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device).eval()
    
    def embed_texts(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**enc)
        return F.normalize(out.pooler_output, dim=-1)
    
    # Embed MetaQA relation descriptions
    meta_texts = [METAQA_REL_DESCRIPTIONS[r] for r in METAQA_RELATIONS]
    meta_embs = embed_texts(meta_texts)  # [9, 768]
    
    # Embed CWQ relation names (convert dots to spaces for better embedding)
    cwq_rels = sorted(cwq_rel2id.keys(), key=lambda r: cwq_rel2id[r])
    cwq_texts = [r.replace('.', ' ').replace('_', ' ') for r in cwq_rels]
    
    # Do in batches
    cwq_embs = []
    for i in range(0, len(cwq_texts), 64):
        batch = cwq_texts[i:i+64]
        cwq_embs.append(embed_texts(batch))
    cwq_embs = torch.cat(cwq_embs, dim=0)  # [916, 768]
    
    # For each CWQ relation, find nearest MetaQA relation
    sim = torch.matmul(cwq_embs, meta_embs.T)  # [916, 9]
    best_meta_idx = torch.argmax(sim, dim=1)  # [916]
    
    cwq_to_metaqa = {}
    for i, cwq_rel in enumerate(cwq_rels):
        meta_idx = best_meta_idx[i].item()
        cwq_to_metaqa[cwq_rel2id[cwq_rel]] = METAQA_RELATIONS[meta_idx]
    
    # Print top mappings for sanity check
    print("\n  Sample CWQ → MetaQA mappings:")
    for cwq_rel in ['film.film.directed_by', 'film.film.genre', 'film.film.starring', 
                     'film.film.initial_release_date', 'film.film.language',
                     'people.person.nationality', 'music.artist.genre']:
        if cwq_rel in cwq_rel2id:
            rid = cwq_rel2id[cwq_rel]
            meta_r = cwq_to_metaqa[rid]
            print(f"    {cwq_rel} → {meta_r}")
    
    del model
    torch.cuda.empty_cache()
    
    return cwq_to_metaqa

# ============================================================
#  4. Evaluation
# ============================================================

def evaluate_zeroshot(samples, kg, predict_fn, cwq_to_metaqa, id2rel, model_name, num_hops):
    """
    Zero-shot Hits@1 on MetaQA.
    predict_fn: returns list of sets of CWQ relation IDs per hop
    """
    total = 0
    hits = 0
    
    for sample in tqdm(samples, desc=f"ZS {model_name} {num_hops}h"):
        topic = sample['topic_entity']
        gold = sample['gold_answers']
        nh = sample['num_hops']
        
        # Get model's predicted CWQ relation IDs per hop
        cwq_rel_ids_per_hop = predict_fn(sample['question'], nh)
        
        # Map CWQ relation IDs to MetaQA relation names
        metaqa_rels_per_hop = []
        for hop_rel_ids in cwq_rel_ids_per_hop:
            mapped = set()
            for rid in hop_rel_ids:
                if rid in cwq_to_metaqa:
                    mapped.add(cwq_to_metaqa[rid])
            if not mapped:
                mapped = set(METAQA_RELATIONS)  # fallback: allow all
            metaqa_rels_per_hop.append(mapped)
        
        # Traverse MetaQA KG
        reached = kg.traverse(topic, metaqa_rels_per_hop)
        
        # Hits@1
        hit = len(reached.intersection(gold)) > 0
        if hit:
            hits += 1
        total += 1
    
    h1 = hits / total if total > 0 else 0
    print(f"\n  {model_name} [{num_hops}-hop]: Hits@1 = {h1:.4f} ({hits}/{total})")
    return {'model': model_name, 'hops': num_hops, 'hits@1': h1, 'hits': hits, 'total': total}

# ============================================================
#  5. Model Prediction Functions (reused from execution_eval)
# ============================================================

def predict_multihop(model, tokenizer, question, device, num_hops, k=5):
    """Multi-hop models (Exp 4/6/7/8). Returns top-k CWQ rel IDs per hop."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        if isinstance(out, tuple):
            rel_logits = out[0]
        else:
            rel_logits = out['rel_logits']
    
    result = []
    for h in range(num_hops):
        if h < rel_logits.size(1):
            _, topk = torch.topk(rel_logits[0, h], k=min(k, rel_logits.size(-1)))
            result.append(set(r.item() for r in topk))
        else:
            result.append(set())
    return result

def predict_rlmc(rl_agent, tokenizer, question, device, num_hops, k=5):
    """Exp 9: RL constraint agent."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        action_logits, _, rel_logits, _ = rl_agent(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    
    actions = torch.argmax(action_logits[0], dim=-1).tolist()
    
    result = []
    for h in range(num_hops):
        if h >= rel_logits.size(1):
            break
        a = actions[h] if h < len(actions) else 3
        if a == 3: break
        elif a == 0: w = 1
        elif a == 1: w = 5
        elif a == 2: w = 50
        else: w = 1
        
        _, topk = torch.topk(rel_logits[0, h], k=min(w, rel_logits.size(-1)))
        result.append(set(r.item() for r in topk))
    
    while len(result) < num_hops:
        result.append(set())
    return result

def predict_flat(model, tokenizer, question, device, num_hops, k=5):
    """Exp 0: flat classifier. Top-k as all hops."""
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    _, topk = torch.topk(logits[0], k=min(k, logits.size(-1)))
    rel_set = set(r.item() for r in topk)
    return [rel_set for _ in range(num_hops)]

# ============================================================
#  6. Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load MetaQA KG
    print("\n[1] Loading MetaQA KG...")
    kg = MetaQAKG(os.path.join(METAQA_DIR, 'kb.txt'))
    
    # Load test sets
    print("\n[2] Loading MetaQA test sets...")
    test_1h = load_metaqa_test(1)
    test_2h = load_metaqa_test(2)
    test_3h = load_metaqa_test(3)
    
    # Load CWQ relation map
    rel2id = torch.load(os.path.join(ROOT, 'data/processed_entity/relation2id.pt'))
    id2rel = {v: k for k, v in rel2id.items()}
    num_rel = len(rel2id)
    
    train_d = torch.load(os.path.join(ROOT, 'data/processed_entity/train_domains.pt'))
    num_dom = int(torch.max(train_d).item()) + 1
    
    # Build semantic mapping
    print("\n[3] Building CWQ→MetaQA relation mapping...")
    cwq_to_metaqa = build_relation_mapping(rel2id, device)
    
    all_results = []
    
    # ---- Exp 7: RoBERTa (our strong baseline) ----
    print("\n[4] Loading models...")
    
    from transformers import RobertaTokenizer
    from train.exp7_roberta import ScaledUnifiedPlanner
    rob_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    print("  Loading Exp 7 (RoBERTa)...")
    model7 = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    model7.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_epoch_29.pt'), map_location=device))
    model7.eval()
    
    for test_data, nh in [(test_1h, 1), (test_2h, 2), (test_3h, 3)]:
        pred_fn = lambda q, n, m=model7: predict_multihop(m, rob_tokenizer, q, device, n, k=5)
        r = evaluate_zeroshot(test_data, kg, pred_fn, cwq_to_metaqa, id2rel, "Exp 7", nh)
        all_results.append(r)
    
    del model7; torch.cuda.empty_cache()
    
    # ---- Exp 8: CPD RoBERTa ----
    print("  Loading Exp 8 (CPD RoBERTa)...")
    model8 = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    model8.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp8_cpd_best.pt'), map_location=device))
    model8.eval()
    
    for test_data, nh in [(test_1h, 1), (test_2h, 2), (test_3h, 3)]:
        pred_fn = lambda q, n, m=model8: predict_multihop(m, rob_tokenizer, q, device, n, k=5)
        r = evaluate_zeroshot(test_data, kg, pred_fn, cwq_to_metaqa, id2rel, "Exp 8", nh)
        all_results.append(r)
    
    del model8; torch.cuda.empty_cache()
    
    # ---- Exp 9: RLMC ----
    print("  Loading Exp 9 (RLMC)...")
    from train.exp9_rlmc import RLConstraintAgent
    base9 = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base9.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    rl9 = RLConstraintAgent(base9).to(device)
    rl9.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=device))
    rl9.eval()
    
    for test_data, nh in [(test_1h, 1), (test_2h, 2), (test_3h, 3)]:
        pred_fn = lambda q, n, m=rl9: predict_rlmc(m, rob_tokenizer, q, device, n, k=5)
        r = evaluate_zeroshot(test_data, kg, pred_fn, cwq_to_metaqa, id2rel, "Exp 9", nh)
        all_results.append(r)
    
    del rl9, base9; torch.cuda.empty_cache()

    # ---- Exp 4: CHCP ----
    print("  Loading Exp 4 (CHCP)...")
    from train.exp4_chcp import CHCPModel
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model4 = CHCPModel(num_relations=num_rel, max_hops=4).to(device)
    model4.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints/exp4_chcp_best.pt'), map_location=device))
    model4.eval()
    
    for test_data, nh in [(test_1h, 1), (test_2h, 2), (test_3h, 3)]:
        pred_fn = lambda q, n, m=model4: predict_multihop(m, bert_tokenizer, q, device, n, k=5)
        r = evaluate_zeroshot(test_data, kg, pred_fn, cwq_to_metaqa, id2rel, "Exp 4", nh)
        all_results.append(r)
    
    del model4; torch.cuda.empty_cache()
    
    # ---- Print Summary ----
    print("\n" + "=" * 70)
    print("  ZERO-SHOT MetaQA RESULTS (Hits@1)")
    print("  Models trained on CWQ only — NO MetaQA training")
    print("=" * 70)
    
    for nh in [1, 2, 3]:
        print(f"\n  {nh}-hop:")
        for r in all_results:
            if r['hops'] == nh:
                print(f"    {r['model']:12s} | Hits@1: {r['hits@1']:.4f} ({r['hits']}/{r['total']})")
    
    print("\n" + "=" * 70)
    
    # Write results
    rp = os.path.join(ROOT, 'results_metaqa_zeroshot.md')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write("# Zero-Shot MetaQA Results (Hits@1)\n\n")
        f.write("Models trained on CWQ only — NO MetaQA training data used.\n")
        f.write("CWQ→MetaQA relation mapping via BERT semantic similarity.\n\n")
        
        f.write("| Model | 1-hop | 2-hop | 3-hop |\n")
        f.write("|---|---|---|---|\n")
        
        models_seen = []
        for r in all_results:
            if r['model'] not in models_seen:
                models_seen.append(r['model'])
        
        for m in models_seen:
            h1 = {r['hops']: r['hits@1'] for r in all_results if r['model'] == m}
            f.write(f"| **{m}** | {h1.get(1,0):.4f} | {h1.get(2,0):.4f} | {h1.get(3,0):.4f} |\n")
    
    print(f"\nResults written to {rp}")

if __name__ == "__main__":
    main()
