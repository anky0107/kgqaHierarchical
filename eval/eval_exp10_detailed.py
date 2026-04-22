import os, sys, torch, functools
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp10_universal import UniversalDataset, collate_universal, UniversalPlanner

def evaluate_dataset(loader, model, device, dataset_name):
    model.eval()
    
    total_samples = 0
    correct_hits1 = 0
    correct_domain = 0
    correct_hop_len = 0
    
    t_bar = tqdm(loader, desc=f"Evaluating {dataset_name}")
    
    for enc, doms, paths, nums, t_ids in t_bar:
        enc = {k: v.to(device) for k, v in enc.items()}
        doms = doms.to(device)
        paths = paths.to(device)
        nums = nums.to(device)
        t_ids = t_ids.to(device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model(enc['input_ids'], enc['attention_mask'], t_ids)
            rel_logits = out['rel_logits']
            dom_logits = out['domain_logits']
            
        B = paths.size(0)
        
        for b in range(B):
            total_samples += 1
            
            # Domain Accuracy
            pred_dom = torch.argmax(dom_logits[b]).item()
            true_dom = doms[b].item()
            if pred_dom == true_dom:
                correct_domain += 1
                
            # Hop Length Accuracy (Can the model predict STOP correctly?)
            # The model predicts up to 3 hops. The true length is nums[b].
            # For simplicity, we check if the exact sequence of predicted relations matches the gold relations up to true length L
            # and if the model stops correctly. 
            L = int(nums[b].item())
            
            path_correct = True
            for h in range(L):
                pred_rel = torch.argmax(rel_logits[b, h]).item()
                true_rel = paths[b, h].item()
                if pred_rel != true_rel:
                    path_correct = False
                    break
                    
            if path_correct:
                correct_hits1 += 1
                
            # Hop length logic: does the model predict the [STOP] equivalent or correctly sequence the exact hops?
            # Since Exp 10 simply generates up to Max Hops, hop length accuracy is implicit in Hits@1 if we consider the exact sequence length.
            # We will approximate Hop Length accuracy by checking if it got the sequence entirely right.
            if path_correct:
                correct_hop_len += 1
                
    dict_res = {
        'hits1': 100 * correct_hits1 / total_samples if total_samples > 0 else 0,
        'domain': 100 * correct_domain / total_samples if total_samples > 0 else 0,
        'hop_len': 100 * correct_hop_len / total_samples if total_samples > 0 else 0
    }
    return dict_res

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Detailed Evaluation (Exp 10) on {device}")
    
    # Load Dicts
    rel_map = torch.load('data/processed_universal/relation2id.pt')
    dom_map = torch.load('data/processed_universal/domain2id.pt')
    num_rel = 861
    num_dom = 71
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    # Init Model
    model = UniversalPlanner(num_dom, num_rel).to(device)
    
    # Load Best Checkpoint weights
    ckpt_path = 'checkpoints/exp10_joint_epoch_4.pt'
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded {ckpt_path} successfully.")
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}")
        print("Evaluating default architecture.")
        
    ds_cwq = UniversalDataset('data/cwq_test.json', rel_map, dom_map, 'cwq')
    ds_webqsp = UniversalDataset('data/processed_universal/webqsp/test.json', rel_map, dom_map, 'webqsp')
    ds_metaqa = UniversalDataset('data/processed_universal/metaqa/test.json', rel_map, dom_map, 'metaqa')
    
    collate = functools.partial(collate_universal, tokenizer=tokenizer)
    
    ldr_cwq = DataLoader(ds_cwq, batch_size=32, collate_fn=collate)
    ldr_webqsp = DataLoader(ds_webqsp, batch_size=32, collate_fn=collate)
    ldr_metaqa = DataLoader(ds_metaqa, batch_size=32, collate_fn=collate)
    
    print("\nStarting Evaluations...")
    res_cwq = evaluate_dataset(ldr_cwq, model, device, 'CWQ')
    res_webqsp = evaluate_dataset(ldr_webqsp, model, device, 'WebQSP')
    res_metaqa = evaluate_dataset(ldr_metaqa, model, device, 'MetaQA')
    
    print("\n" + "="*50)
    print("EXPERIMENT 10: METRICS AND STATISTICS (Joint Tagging)")
    print("="*50)
    
    print("\n--- ComplexWebQuestions (CWQ) ---")
    print(f"Hits@1 (Path Exact Match) : {res_cwq['hits1']:.2f}%")
    print(f"Domain Prediction Acc     : {res_cwq['domain']:.2f}%")
    
    print("\n--- WebQSP ---")
    print(f"Hits@1 (Path Exact Match) : {res_webqsp['hits1']:.2f}%")
    print(f"Domain Prediction Acc     : {res_webqsp['domain']:.2f}%")
    
    print("\n--- MetaQA ---")
    print(f"Hits@1 (Path Exact Match) : {res_metaqa['hits1']:.2f}%")
    print(f"Domain Prediction Acc     : {res_metaqa['domain']:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
