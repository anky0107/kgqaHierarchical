import os, sys, json, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

"""
precompute_entity_embs.py — CDS Pipeline: F1 Entity Embedding Cache
====================================================================

Paper Section: §V-C  "Cascading Dual-Stage Filtering (CDS) — F1 Bi-Encoder"

Purpose
-------
Precomputes and caches a dense embedding for every Freebase entity (MID) in
the knowledge graph using the Stage-1 bi-encoder (MiniLM-L6-v2).  The
resulting embedding matrix is consumed by the F1 nearest-neighbour retrieval
step that narrows the full entity space down to the top-200 candidates before
handing off to the MPNet F2 ranker.

By precomputing these embeddings offline we avoid re-encoding the entire KG
at every inference call — a critical efficiency optimisation given that the
master_mid2name dictionary typically contains millions of entities.

Pipeline position
-----------------
  [master_mid2name.json]  ←  all KG entities + their textual names
        │
        ▼
  [precompute_entity_embs.py]  ← THIS FILE
        │  produces: data/exp16_entity_embs.pt
        ▼
  [F1 retrieval at inference]  cosine(question_emb, entity_embs) → top-200

Inputs
------
- data/master_mid2name.json         : {mid: human_readable_name, ...}
- checkpoints/exp16_s1_bi.pt        : (optional) fine-tuned S1 bi-encoder
                                       weights; falls back to HuggingFace
                                       weights if the checkpoint is absent.

Outputs
-------
- data/exp16_entity_embs.pt  : {
      'mids' : List[str],     # ordered list of Freebase MIDs
      'embs' : torch.Tensor   # shape [N_entities, 384]  (float32, CPU)
  }

Key hyperparameters
-------------------
- model_name : "sentence-transformers/all-MiniLM-L6-v2"  (F1 encoder, §V-C)
- batch_size : 512  (GPU batch for embedding inference)
- max_length : 64   (token truncation for entity name strings)
- Embedding  : CLS-token representation (index 0 of last hidden state)

How it works
------------
1. Load the MiniLM-L6-v2 model.  If a fine-tuned S1 checkpoint exists, inject
   those weights so the entity embeddings are aligned with the question encoder
   used during retrieval.
2. Iterate over all entity names in batches of 512, encoding each with the
   model and extracting the CLS token embedding.
3. Concatenate all batch embeddings into a single [N, 384] tensor and save
   together with the corresponding MID list so downstream code can map
   indices back to entity identifiers.
"""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Main precomputation routine
# ──────────────────────────────────────────────────────────────────────────────

def precompute():
    """
    Encode all KG entity names with the S1 bi-encoder and persist the result.

    The function is intentionally structured as a single linear pass:
      load model → load entity names → batch-encode → save tensor.
    No training occurs here; the model is used in eval/inference mode only.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Load the S1 bi-encoder ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)

    # Optionally inject fine-tuned weights from the S1 training run (exp16).
    # This aligns entity embeddings with the question encoder, improving
    # cosine-similarity recall at the F1 retrieval stage.
    s1_path = os.path.join(ROOT, 'checkpoints/exp16_s1_bi.pt')
    if os.path.exists(s1_path):
        print(f"Loading custom S1 weights from {s1_path}")
        model.load_state_dict(torch.load(s1_path, map_location=device))
    model.eval()  # inference only — disable dropout

    # ── Load entity name dictionary ────────────────────────────────────────
    print("Loading Master MID2Name...")
    mid2name = json.load(open(os.path.join(ROOT, 'data/master_mid2name.json'), 'r', encoding='utf-8'))
    # Preserve insertion order so that index ↔ MID correspondence is stable
    mids  = list(mid2name.keys())
    names = [mid2name[m] for m in mids]   # parallel list of human-readable names

    # ── Batch-encode all entity names ──────────────────────────────────────
    batch_size = 512   # chosen to saturate GPU VRAM without OOM on typical cards
    all_embs   = []

    print(f"Embedding {len(names)} entities...")
    with torch.no_grad():
        for i in tqdm(range(0, len(names), batch_size)):
            batch = names[i:i+batch_size]
            # Tokenise with padding/truncation; max_length=64 is sufficient for
            # entity names which are typically short noun phrases.
            enc  = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
            # Extract CLS token (index 0) from the last hidden state.
            # For sentence-transformers models trained with mean pooling this is
            # a reasonable approximation; the fine-tuned S1 checkpoint was also
            # trained with CLS extraction to stay consistent.
            embs = model(**enc).last_hidden_state[:, 0, :]  # CLS
            all_embs.append(embs.cpu())   # move to CPU immediately to free VRAM

    # Concatenate along the entity dimension → [N_entities, hidden_dim]
    all_embs = torch.cat(all_embs, dim=0)

    # ── Persist embedding cache ────────────────────────────────────────────
    # Saving mids alongside embs is crucial: downstream retrieval code needs to
    # map from ranked indices back to Freebase MID strings.
    print("Saving embeddings...")
    torch.save({
        'mids': mids,
        'embs': all_embs
    }, os.path.join(ROOT, 'data/exp16_entity_embs.pt'))
    print("Done!")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    precompute()
