import os, sys, torch, json, functools
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp10_universal import UniversalPlanner, DATASET_IDS
# We use Untagged from exp11 so that the text prompt doesn't have the dataset name
from train.exp11_vector_projection import UntaggedUniversalDataset, collate_untagged

def evaluate_blind(model, loader, device, dataset_name, id2rel):
    model.eval()
    
    # Tracking by hop count
    total_by_hops = defaultdict(int)
    correct_by_hops = defaultdict(int)
    
    t_bar = tqdm(loader, desc=f"Blind Eval {dataset_name}")
    
    for enc, paths, nums in t_bar:
        enc = enc.to(device); paths = paths.to(device)
        B = paths.size(0)
        
        best_preds = []
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            # We run the model 3 times, once for each possible latent dataset tag (0, 1, 2)
            # We then select the prediction that the model was MOST confident about.
            max_confidences = torch.zeros(B, device=device) - 1
            selected_preds = torch.zeros((B, 4), dtype=torch.long, device=device)
            
            for tag_id in [0, 1, 2]:
                ds_ids = torch.full((B,), tag_id, dtype=torch.long, device=device)
                out = model(enc['input_ids'], enc['attention_mask'], ds_ids)
                
                # Softmax over relations to get confidence
                probs = F.softmax(out['rel_logits'], dim=-1) # [B, H, num_rel]
                
                # Get the max prob at each hop: [B, H]
                hop_max_probs, hop_preds = torch.max(probs, dim=-1)
                
                # To get sequence confidence, we multiply the probs of the hops
                # (We only look at the valid number of hops for the true sequence length for this evaluation,
                # or just look at hop 1 to decide the tag since tags heavily dictate the first relation)
                seq_confidences = hop_max_probs[:, 0] 
                
                # Update best predictions
                for b in range(B):
                    if seq_confidences[b] > max_confidences[b]:
                        max_confidences[b] = seq_confidences[b]
                        selected_preds[b] = hop_preds[b]
                        
        # Now score the best selected predictions
        for b in range(B):
            L = int(nums[b].item())
            target_path = paths[b, :L].tolist()
            pred_path = selected_preds[b, :L].tolist()
            
            total_by_hops[L] += 1
            if target_path == pred_path:
                correct_by_hops[L] += 1
                
        t_bar.set_postfix(acc=f"{100*sum(correct_by_hops.values())/sum(total_by_hops.values()):.2f}%")
        
    print(f"\n[Result] BLIND {dataset_name} Breakdown:")
    for hops in sorted(total_by_hops.keys()):
        acc = correct_by_hops[hops] / total_by_hops[hops]
        print(f"  {hops}-Hop Accuracy: {100*acc:.2f}% ({correct_by_hops[hops]}/{total_by_hops[hops]})")
        
    overall_acc = sum(correct_by_hops.values()) / sum(total_by_hops.values())
    print(f"  Overall Score  : {100*overall_acc:.2f}%\n")
    return overall_acc, correct_by_hops, total_by_hops

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join(ROOT, 'data/processed_universal')
    
    relation2id = torch.load(os.path.join(proc_dir, 'relation2id.pt'))
    domain2id = torch.load(os.path.join(proc_dir, 'domain2id.pt'))
    id2rel = {v: k for k, v in relation2id.items()}
    
    num_rel = max(relation2id.values()) + 1
    num_dom = max(domain2id.values()) + 1
    
    model = UniversalPlanner(num_dom, num_rel).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    # domain head matching
    ckpt_num_dom = state_dict['domain_head.weight'].shape[0] if 'domain_head.weight' in state_dict else num_dom
    if ckpt_num_dom != num_dom:
        model.domain_head.weight.data[:ckpt_num_dom] = state_dict['domain_head.weight']
        model.domain_head.bias.data[:ckpt_num_dom] = state_dict['domain_head.bias']
        del state_dict['domain_head.weight']
        del state_dict['domain_head.bias']
        
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint for BLIND eval: {args.ckpt}")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    collate = functools.partial(collate_untagged, tokenizer=tokenizer)
    
    # Datasets
    cwq_ds = UntaggedUniversalDataset(os.path.join(proc_dir, 'cwq/dev.json'), relation2id)
    webq_ds = UntaggedUniversalDataset(os.path.join(proc_dir, 'webqsp/test.json'), relation2id)
    meta_ds = UntaggedUniversalDataset(os.path.join(proc_dir, 'metaqa/test.json'), relation2id)
    
    cwq_loader = DataLoader(cwq_ds, batch_size=32, collate_fn=collate)
    webq_loader = DataLoader(webq_ds, batch_size=32, collate_fn=collate)
    meta_loader = DataLoader(meta_ds, batch_size=64, collate_fn=collate)
    
    print("\n" + "="*50)
    print("STARTING STRICT BLIND EVALUATION (NO DATASET TAGS)")
    print("="*50)
    
    evaluate_blind(model, cwq_loader, device, 'CWQ-Dev', id2rel)
    evaluate_blind(model, webq_loader, device, 'WebQSP-Test', id2rel)
    evaluate_blind(model, meta_loader, device, 'MetaQA-Test', id2rel)

if __name__ == '__main__':
    main()
