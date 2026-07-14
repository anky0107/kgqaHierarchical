import os, sys, json, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def precompute():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Load weights from our S1 Bi-Encoder
    s1_path = os.path.join(ROOT, 'checkpoints/exp16_s1_bi.pt')
    if os.path.exists(s1_path):
        print(f"Loading custom S1 weights from {s1_path}")
        model.load_state_dict(torch.load(s1_path, map_location=device))
    model.eval()

    print("Loading Master MID2Name...")
    mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
    mids = list(mid2name.keys())
    names = [mid2name[m] for m in mids]
    
    batch_size = 512
    all_embs = []
    
    print(f"Embedding {len(names)} entities...")
    with torch.no_grad():
        for i in tqdm(range(0, len(names), batch_size)):
            batch = names[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
            embs = model(**enc).last_hidden_state[:, 0, :] # CLS
            all_embs.append(embs.cpu())
            
    all_embs = torch.cat(all_embs, dim=0)
    
    # Save mids and embs
    print("Saving embeddings...")
    torch.save({
        'mids': mids,
        'embs': all_embs
    }, os.path.join(ROOT, 'data/exp16_entity_embs.pt'))
    print("Done!")

if __name__ == "__main__":
    precompute()
