import os, sys, json, torch, functools
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ============================================================
#  Seq2Seq Dataset Loader (No Tags, Target is Textual Path)
# ============================================================
class Seq2SeqUniversalDataset(Dataset):
    def __init__(self, json_path, max_hops=4):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            rels = item.get('relations', [])
            if not rels: continue
            
            # Target generation: "relation1 | relation2 | relation3"
            target_path = " | ".join(rels[:max_hops])
            
            # Input generation
            question = item.get('question', '')
            topic = item.get('topic_entity', '')
            question_with_topic = f"parse: topic: {topic} question: {question}" if topic else f"parse: {question}"
            
            self.samples.append({
                'input': question_with_topic,
                'target': target_path
            })
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_seq2seq(batch, tokenizer):
    inputs = [b['input'] for b in batch]
    targets = [b['target'] for b in batch]
    
    enc_in = tokenizer(inputs, padding=True, truncation=True, max_length=160, return_tensors='pt')
    enc_tgt = tokenizer(targets, padding=True, truncation=True, max_length=128, return_tensors='pt')
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    labels = enc_tgt.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    
    return enc_in, labels

# ============================================================
#  Exp 12 Model: Seq2Seq Translator (NS-KGQA style)
# ============================================================
class Seq2SeqTranslator(nn.Module):
    def __init__(self, model_name="t5-base"):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss
        
    def generate(self, input_ids, attention_mask, max_length=128):
        return self.t5.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

def train_exp12_seq2seq():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join(ROOT, 'data/processed_universal')

    # 1. Datasets & Loaders
    tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
    collate = functools.partial(collate_seq2seq, tokenizer=tokenizer)
    
    cwq_train_ds = Seq2SeqUniversalDataset(os.path.join(proc_dir, 'cwq/train.json'))
    webq_train_ds = Seq2SeqUniversalDataset(os.path.join(proc_dir, 'webqsp/train.json'))
    meta_train_ds = Seq2SeqUniversalDataset(os.path.join(proc_dir, 'metaqa/train.json'))

    datasets = [cwq_train_ds, webq_train_ds, meta_train_ds]
    combined_ds = ConcatDataset(datasets)
    
    weights = []
    for ds in datasets:
        ds_weight = 1.0 / len(ds) / len(datasets)
        weights.extend([ds_weight] * len(ds))
    
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)
    joint_loader = DataLoader(combined_ds, batch_size=8, sampler=sampler, collate_fn=collate, num_workers=2)

    # 2. Model & Optimizer
    model = Seq2SeqTranslator().to(device)
    model.t5.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # T5 uses higher LR
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    
    # 3. Train Loop
    print("\nStarting Exp 12: Seq2Seq Translator (Abstract Planning)")
    for epoch in range(5):
        model.train()
        t_bar = tqdm(joint_loader, desc=f"Ep {epoch} Seq2Seq")
        
        for i, (enc_in, labels) in enumerate(t_bar):
            input_ids = enc_in['input_ids'].to(device)
            attention_mask = enc_in['attention_mask'].to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda'):
                loss = model(input_ids, attention_mask, labels)
                loss = loss / 4 # Accumulation
                
            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            t_bar.set_postfix(loss=loss.item() * 4)
            
        torch.save(model.state_dict(), os.path.join(ROOT, f'checkpoints/exp12_seq2seq_epoch_{epoch}.pt'))

if __name__ == '__main__':
    train_exp12_seq2seq()
