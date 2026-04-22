import os, sys, torch, torch.nn as nn, torch.nn.functional as F
from transformers import RobertaTokenizer

# Fix Windows encoding issues
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from train.exp7_roberta import ScaledUnifiedPlanner
from train.exp9_rlmc import RLConstraintAgent

def trace_inference(question):
    print(f"\n{'='*60}")
    print(f" EXPERIMENT 9 INFERENCE TRACE: {question}")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Loading Maps and Model
    print(f"[STAGE 0] Loading Knowledge Graph Maps and Checkpoints...")
    rel2id = torch.load('data/processed_entity/relation2id.pt', map_location='cpu')
    id2rel = {v: k for k, v in rel2id.items()}
    dom2id = torch.load('data/processed_entity/domain2id.pt', map_location='cpu')
    num_rel = len(rel2id); num_dom = len(dom2id)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    base_model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    base_model.load_state_dict(torch.load('checkpoints/exp7_roberta_best.pt', map_location=device))
    
    rl_agent = RLConstraintAgent(base_model).to(device)
    rl_checkpoint = 'checkpoints/exp9_rlmc_epoch_9.pt'
    if os.path.exists(rl_checkpoint):
        rl_agent.load_state_dict(torch.load(rl_checkpoint, map_location=device))
        print("  -> RL Agent loaded from checkpoint.")
    else:
        print("  -> WARNING: No RL checkpoint found. Using initial weights.")
    
    rl_agent.eval()
    
    # 2. Tokenization
    print(f"\n[STAGE 1] Tokenization (BPE Strategy)")
    tokens = tokenizer.tokenize(question)
    print(f"  Tokens: {tokens}")
    enc = tokenizer(question, return_tensors='pt').to(device)
    input_ids = enc['input_ids']
    print(f"  Input IDs Shape: {input_ids.shape} (Includes [CLS] and [SEP])")

    # 3. Embedding Layer (Step-by-Step inside RoBERTa)
    print(f"\n[STAGE 2] RoBERTa Embedding Layer (Static Dictionary)")
    with torch.no_grad():
        # Accessing the internal embedding layer of the roberta model
        embeddings = rl_agent.base_model.encoder.embeddings.word_embeddings(input_ids)
        print(f"  Embedding Shape: {embeddings.shape} ([Batch, Tokens, 1024])")
        print(f"  Mean Vector Val : {embeddings.mean().item():.4f}")
        print(f"  Vector Norm     : {torch.norm(embeddings).item():.4f}")

    # 4. Semantic Encoder (24 Transformer Layers)
    print(f"\n[STAGE 3] Semantic Encoder (Contextualizing...)")
    with torch.no_grad():
        outputs = rl_agent.base_model.encoder(input_ids, enc['attention_mask'])
        last_hidden = outputs.last_hidden_state
        print(f"  Contextual Output Shape: {last_hidden.shape} ([Batch, Tokens, 1024])")
        
        # Extract CLS token vector
        q_h = last_hidden[:, 0, :]
        print(f"  [CLS] token vector extracted. Size: {q_h.shape}")
        print(f"  [CLS] Contextual Norm: {torch.norm(q_h).item():.4f}")

    # 5. Dimensionality Squeeze (The Bottleneck)
    print(f"\n[STAGE 4] Feature Distillation (1024 -> 512 Linear Squeeze)")
    with torch.no_grad():
        h_q = rl_agent.base_model.proj(q_h)
        print(f"  Compressed vector shape: {h_q.shape}")
        print(f"  Squeezed Norm: {torch.norm(h_q).item():.4f}")

    # 6. Hop Embedding Injection (Broadcast Addition)
    print(f"\n[STAGE 5] Hop Injection (Broadcasting 4 unique 'ID Tags')")
    with torch.no_grad():
        # Broadcast h_q across 4 hop slots
        broadcasted_q = h_q.unsqueeze(1) # [1, 1, 512]
        # Add the learnable hop embeddings
        init_repr = broadcasted_q + rl_agent.base_model.hop_embeddings.unsqueeze(0)
        print(f"  Input to Reasoning Transformer shape: {init_repr.shape} ([Batch, 4, 512])")
        for h in range(4):
            hnorm = torch.norm(init_repr[0, h]).item()
            print(f"    - Slot {h+1} injected norm: {hnorm:.4f}")

    # 7. Reasoning Transformer (4-Layer Coherence)
    print(f"\n[STAGE 6] Reasoning Transformer (Self-Attention across Hops)")
    with torch.no_grad():
        refined_repr = rl_agent.base_model.transformer(init_repr)
        print(f"  Refined Intent Repr shape: {refined_repr.shape}")
        
        # Let's see the Relation candidates for the hops
        rel_logits = rl_agent.base_model.relation_head(refined_repr)

    # 8. RL Decision (The Meta-Constraint Head)
    print(f"\n[STAGE 7] RL Meta-Constraint Agent Decision")
    with torch.no_grad():
        action_logits = rl_agent.policy_head(refined_repr)
        probs = F.softmax(action_logits, dim=-1)[0]
        
        actions_map = {0: "TIGHT (K=1)", 1: "MEDIUM (K=5)", 2: "LOOSE (K=50)", 3: "STOP (Prune)"}
        
        for h in range(4):
            # Top relation for visualization
            top_rel_id = torch.argmax(rel_logits[0, h]).item()
            rel_name = id2rel.get(top_rel_id, "Unknown")
            
            # RL Probabilities
            h_probs = probs[h]
            best_action = torch.argmax(h_probs).item()
            
            print(f"  HOP {h+1}:")
            print(f"    Top Predicted Relation: {rel_name}")
            print(f"    RL Action Probabilities:")
            for a_idx, a_name in actions_map.items():
                print(f"       - {a_name:15s}: {h_probs[a_idx].item()*100:5.2f}%")
            print(f"    FINAL DECISION: {actions_map[best_action]}")

    print(f"\n{'='*60}")
    print(f" TRACE COMPLETE.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="Who is the mascot of the team Lou Seal represents?")
    args = parser.parse_args()
    trace_inference(args.question)
