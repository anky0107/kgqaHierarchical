import os, sys, json, torch, functools
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ============================================================
#  Subgraph Reader Dataset Loader
# ============================================================
class SubgraphReaderDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            question = item.get('question', '')
            context = item.get('subgraph_context', '')
            if not context: continue
            
            # Since we don't have exact char spans for QA training right now,
            # we will train the reader generically as a sequence classifier first
            # to verify pipeline health, before adding complex start/end index extraction.
            # In a real QA setup, we need exact char offsets in the context.
            rel_len = len(item.get('relations', []))
            
            self.samples.append({
                'question': question,
                'context': context,
                'num_hops': rel_len
            })
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_reader(batch, tokenizer):
    questions = [b['question'] for b in batch]
    contexts = [b['context'] for b in batch]
    
    enc = tokenizer(questions, contexts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Dummy start/end positions for pipeline completion
    bsz = len(batch)
    start_positions = torch.ones(bsz, dtype=torch.long)
    end_positions = torch.ones(bsz, dtype=torch.long) * 2
    
    return enc, start_positions, end_positions

# ============================================================
#  Exp 13 Model: Subgraph Prompting Reader
# ============================================================
def train_exp13_subgraph_reader():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join(ROOT, 'data/processed_universal')

    # Using RoBERTa Base as our "Reader" for now (Longformer would be better for huge graphs)
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collate = functools.partial(collate_reader, tokenizer=tokenizer)
    
    cwq_train_ds = SubgraphReaderDataset(os.path.join(proc_dir, 'cwq/train_with_subgraphs.json'))
    webq_train_ds = SubgraphReaderDataset(os.path.join(proc_dir, 'webqsp/train_with_subgraphs.json'))
    meta_train_ds = SubgraphReaderDataset(os.path.join(proc_dir, 'metaqa/train_with_subgraphs.json'))

    datasets = [cwq_train_ds, webq_train_ds, meta_train_ds]
    combined_ds = ConcatDataset(datasets)
    
    weights = []
    for ds in datasets:
        ds_weight = 1.0 / len(ds) / len(datasets)
        weights.extend([ds_weight] * len(ds))
    
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)
    joint_loader = DataLoader(combined_ds, batch_size=16, sampler=sampler, collate_fn=collate, num_workers=2)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    
    print(f"\nStarting Exp 13: Subgraph Prompting Reader (Data: {len(combined_ds)} samples)")
    for epoch in range(5):
        model.train()
        t_bar = tqdm(joint_loader, desc=f"Ep {epoch} RC Reader")
        
        for i, (enc, starts, ends) in enumerate(t_bar):
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            starts = starts.to(device)
            ends = ends.to(device)
            
            with torch.amp.autocast('cuda'):
                # Forward pass QA loss
                output = model(input_ids=input_ids, attention_mask=attention_mask, 
                               start_positions=starts, end_positions=ends)
                loss = output.loss / 2
                
            scaler.scale(loss).backward()
            if (i + 1) % 2 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            t_bar.set_postfix(loss=loss.item() * 2)
            
        torch.save(model.state_dict(), os.path.join(ROOT, f'checkpoints/exp13_reader_epoch_{epoch}.pt'))

if __name__ == '__main__':
    train_exp13_subgraph_reader()
