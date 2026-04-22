"""
Evaluate Exp 9 RLMC using the SAME path-matching protocol as Exp 0-8.
This gives directly comparable Hits@1 / Hits@3 / Hop Accuracy numbers.
"""
import os, sys, torch, json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from transformers import RobertaTokenizer
from train.exp7_roberta import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent
from eval.e2e_evaluate import extract_evaluation_data, evaluate_model, predict_exp9

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load relation maps
    data_dir = os.path.join(ROOT, 'data/processed_entity')
    relation2id = torch.load(os.path.join(data_dir, 'relation2id.pt'))
    id2relation = {v: k for k, v in relation2id.items()}
    
    train_d = torch.load(os.path.join(data_dir, 'train_domains.pt'))
    num_dom = int(torch.max(train_d).item()) + 1
    num_rel = len(relation2id)

    # Extract evaluation data
    print("\nExtracting evaluation data...")
    dev_data = json.load(open(os.path.join(ROOT, 'data/cwq_dev.json'), 'r', encoding='utf-8'))
    test_data = json.load(open(os.path.join(ROOT, 'data/cwq_test.json'), 'r', encoding='utf-8'))
    
    print("  Dev set:")
    dev_samples = extract_evaluation_data(dev_data, relation2id)
    print("  Test set:")
    test_samples = extract_evaluation_data(test_data, relation2id)

    datasets = [("Dev", dev_samples), ("Test", test_samples)]

    # Load Exp 9 RLMC
    print("\n  ---- Exp 9: RL Meta-Constraint Agent (RLMC) ----")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base_model.load_state_dict(torch.load(
        os.path.join(ROOT, 'checkpoints/exp7_roberta_best.pt'), map_location=device))
    
    rl_model = RLConstraintAgent(base_model).to(device)
    rl_model.load_state_dict(torch.load(
        os.path.join(ROOT, 'checkpoints/exp9_rlmc_epoch_9.pt'), map_location=device))
    rl_model.eval()

    all_results = []
    for s_name, s_data in datasets:
        r9 = evaluate_model(
            s_data, rl_model, tokenizer, id2relation, device,
            f"Exp 9 ({s_name})", predict_exp9, model_type='rlmc')
        all_results.append(r9)

    print("\n" + "=" * 60)
    print("  EXP 9 PATH-MATCHING RESULTS (same protocol as Exp 0-8)")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['model']:30s} | H@1: {r['hits@1']:.4f} | H@3: {r['hits@3']:.4f} | Hop: {r['hop_accuracy']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
