import os, sys, json, torch
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference_pipeline.exp7_optimized import Exp7Optimized
from inference_pipeline.exp9_optimized import Exp9Optimized
from utils.sparql_parser import find_reasoning_path

def load_train_data(limit=100):
    train_path = os.path.join(ROOT, 'data/cwq_train.json')
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        question = item['question']
        gold_answers = [a['answer_id'] for a in item['answers']]
        sparql = item['sparql']
        
        # Extract topic entity from SPARQL
        path = find_reasoning_path(sparql)
        if not path:
            continue
            
        topic_entity = path[0][0]
        # path[0][0] might be 'ns:m.xxxx'
        if topic_entity.startswith("ns:"):
            topic_entity = topic_entity[3:]
            
        samples.append({
            'question': question,
            'topic_entity': topic_entity,
            'gold_answers': set(gold_answers)
        })
        
        if len(samples) >= limit:
            break
            
    return samples

def evaluate():
    print("Initializing Models...")
    exp7 = Exp7Optimized()
    exp9 = Exp9Optimized()
    
    limit = 200 # Starting with 200 samples
    print(f"Loading {limit} samples from train set...")
    samples = load_train_data(limit=limit)
    
    results = {
        'exp7': {'hits': 0, 'total_entities': 0, 'count': 0},
        'exp9': {'hits': 0, 'total_entities': 0, 'count': 0}
    }
    
    # Silence prints during inference
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        for i, sample in enumerate(tqdm(samples, desc="Evaluating", file=original_stdout)):
            q = sample['question']
            te = sample['topic_entity']
            gold = sample['gold_answers']
            
            # --- Exp 7 ---
            try:
                ans7 = exp7.run_inference(q, te)
                if any(a in gold for a in ans7):
                    results['exp7']['hits'] += 1
                results['exp7']['total_entities'] += len(ans7)
                results['exp7']['count'] += 1
            except Exception as e:
                pass
                
            # --- Exp 9 ---
            try:
                ans9 = exp9.run_inference(q, te)
                if any(a in gold for a in ans9):
                    results['exp9']['hits'] += 1
                results['exp9']['total_entities'] += len(ans9)
                results['exp9']['count'] += 1
            except Exception as e:
                pass
    finally:
        sys.stdout = original_stdout

    print("\n" + "="*40)
    print("EVALUATION RESULTS (Train Set Subset)")
    print("="*40)
    
    for name in ['exp7', 'exp9']:
        res = results[name]
        if res['count'] > 0:
            success_rate = (res['hits'] / res['count']) * 100
            avg_entities = res['total_entities'] / res['count']
            print(f"[{name.upper()}]")
            print(f"  Success (Hit@N): {success_rate:.2f}%")
            print(f"  Avg Entities:   {avg_entities:.2f}")
            print(f"  Total Samples:  {res['count']}")
            print("-" * 20)
            
if __name__ == "__main__":
    evaluate()
