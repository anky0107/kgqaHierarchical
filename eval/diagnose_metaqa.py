import torch, os, sys
from transformers import RobertaTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from train.exp7_roberta import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent

def diagnose():
    device = torch.device('cuda')
    # Load IDs
    rel2id = torch.load('data/processed_entity/relation2id.pt')
    id2rel = {v: k for k, v in rel2id.items()}
    dom2id = torch.load('data/processed_entity/domain2id.pt')
    num_dom = len(dom2id)
    num_rel = len(rel2id)

    # Load Exp 9
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    base = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base.load_state_dict(torch.load('checkpoints/exp7_roberta_best.pt', map_location=device))
    rl9 = RLConstraintAgent(base).to(device)
    rl9.load_state_dict(torch.load('checkpoints/exp9_rlmc_epoch_9.pt', map_location=device))
    rl9.eval()

    test_q = [
        "who directed [Kismet]",
        "who wrote the movie [Kismet]",
        "which actors star in [Kismet]",
        "[Marlene Dietrich] appears in which movies",
        "the films directed by [William Dieterle] were released in which years",
        "what genre is the movie [Kismet]"
    ]

    print(f"{'Question':<50} | {'Top-1 Rel (CWQ)':<40} | {'Action'}")
    print("-" * 110)

    for q in test_q:
        q_clean = q.replace('[', '').replace(']', '')
        enc = tokenizer(q_clean, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            action_logits, _, rel_logits, _ = rl9(enc['input_ids'], enc['attention_mask'])
        
        # Predicted Rel
        top_rel_id = torch.argmax(rel_logits[0, 0]).item()
        top_rel_name = id2rel[top_rel_id]
        
        # Action
        action = torch.argmax(action_logits[0, 0]).item()
        action_names = ["TIGHT(1)", "MEDIUM(5)", "LOOSE(50)", "STOP"]
        
        print(f"{q:<50} | {top_rel_name:<40} | {action_names[action]}")

if __name__ == "__main__":
    diagnose()
