"""
models.py — Model definitions and checkpoint loaders for the CDS pipeline.

Each load_* function returns (tokenizer, model) ready for inference.
The PathAwareRanker defined here is the canonical version — use this
instead of the copies scattered across the train/ scripts.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    T5EncoderModel,
)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT    = os.path.join(ROOT, "checkpoints")

# ─────────────────────────────────────────────────────────────
#  Stage 2 model
# ─────────────────────────────────────────────────────────────

class PathAwareRanker(nn.Module):
    """
    Stage 2: MPNet-base-v2 shared encoder + 3-input MLP fusion head.

    Architecture (verified from exp16v2_train.py):
        q_emb, p_emb, e_emb = encoder(q), encoder(p), encoder(e)   # 768-dim each
        score = MLP(concat[q_emb, p_emb, e_emb])                   # → scalar

    NOTE: The inference code in the original benchmark scripts bypassed
    this MLP and used raw cosine_similarity(q_emb + p_emb, e_emb).
    This class restores the correct forward pass.
    """

    ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(self.ENCODER_NAME)
        hidden = self.encoder.config.hidden_size          # 768
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        q_ids: torch.Tensor, q_mask: torch.Tensor,
        p_ids: torch.Tensor, p_mask: torch.Tensor,
        e_ids: torch.Tensor, e_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.encoder
        q = enc(q_ids,  attention_mask=q_mask).last_hidden_state[:, 0, :]
        p = enc(p_ids,  attention_mask=p_mask).last_hidden_state[:, 0, :]
        e = enc(e_ids,  attention_mask=e_mask).last_hidden_state[:, 0, :]
        return self.fuse(torch.cat([q, p, e], dim=-1)).squeeze(-1)


class GraphAwareRanker(nn.Module):
    """
    Stage 2: MPNet-base-v2 shared encoder + Graph Features + MLP fusion head.
    """
    ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, num_graph_features: int = 3) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(self.ENCODER_NAME)
        hidden = self.encoder.config.hidden_size  # 768
        
        # 3 * 768 (q, p, e) + graph_features
        mlp_input_dim = (hidden * 3) + num_graph_features
        
        self.fuse = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        q_ids: torch.Tensor, q_mask: torch.Tensor,
        p_ids: torch.Tensor, p_mask: torch.Tensor,
        e_ids: torch.Tensor, e_mask: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.encoder
        q = enc(q_ids, attention_mask=q_mask).last_hidden_state[:, 0, :]
        p = enc(p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :]
        e = enc(e_ids, attention_mask=e_mask).last_hidden_state[:, 0, :]
        
        # Concatenate text embeddings with structural graph features
        combined = torch.cat([q, p, e, graph_features], dim=-1)
        
        return self.fuse(combined).squeeze(-1)


class PureRLPolicy(nn.Module):
    """
    Stage 3: Pure RL Feature-Based Policy Network.
    Takes lightweight hand-crafted features instead of raw text embeddings.
    """
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape [..., 5]
        Returns: logits of shape [...]
        """
        return self.net(features).squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────

def load_stage1(device: torch.device):
    """
    Stage 1 — Bi-Encoder (MiniLM-L6-v2, 22M params).
    Checkpoint: checkpoints/exp16v2_s1_bi.pt
    """
    name = "sentence-transformers/all-MiniLM-L6-v2"
    ckpt = os.path.join(CKPT, "exp16v2_s1_bi.pt")
    _require(ckpt)
    tok   = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"[S1] Loaded MiniLM-L6-v2  <- {os.path.basename(ckpt)}")
    return tok, model


def load_stage2(device: torch.device, version: str = "v1"):
    """
    Stage 2 — PathAwareRanker (MPNet-base-v2 + MLP, 109M params).

    version='v1' : exp16v2_s2_path.pt   — SoftMargin loss (original)
    version='v2' : exp25_s2_listwise.pt — KL-Distillation Listwise loss (Exp 25)

    BUG FIX: previous benchmark scripts loaded only the base MPNet encoder
    and called F.cosine_similarity(q+p, e), bypassing the trained MLP
    fusion head entirely. This loader returns the full PathAwareRanker
    with the MLP weights correctly restored.
    """
    if version == "v3_gnn":
        ckpt_name = "exp29_s2_gnn_ranker.pt"
    elif version == "v2":
        ckpt_name = "exp25_s2_listwise.pt"
    else:
        ckpt_name = "exp16v2_s2_path.pt"
    ckpt = os.path.join(CKPT, ckpt_name)
    _require(ckpt)
    tok   = AutoTokenizer.from_pretrained(PathAwareRanker.ENCODER_NAME)
    
    if version == "v3_gnn":
        model = GraphAwareRanker().to(device)
    else:
        model = PathAwareRanker().to(device)
        
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"[S2] Loaded Ranker ({version})  <- {os.path.basename(ckpt)}")
    return tok, model


def load_stage3(device: torch.device, version: str = "v2"):
    """
    Stage 3 — BGE cross-encoder (109M params).

    version='v2' : exp16v2_s3_cross.pt        — (question, entity_name)
    version='v3' : exp16v3_s3_cross.pt        — (question [PATH] path, entity_name)
    version='v4' : exp17_s3_enriched.pt       — (question, name | path_nl | type)
    version='v5' : exp19_s3_hard_negatives.pt — v4 trained on 2k hard negatives
    version='v6' : exp23_s3_full_hard_neg.pt  — v4 trained on full 27k hard negatives (Exp 23)
    version='v7' : exp24_s3_path_v7.pt        — (question, name | path_nl)  [Exp 24, no type]
    """
    name      = "BAAI/bge-reranker-base"
    if version == "v10_pure_rl":
        ckpt_path = os.path.join(ROOT, "checkpoints", "exp28_pure_rl_s3.pt")
        print(f"[S3] Loading Pure Feature-Based RL Policy <- {os.path.basename(ckpt_path)}")
        model = PureRLPolicy().to(device)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"     [WARNING] Checkpoint not found! Initializing randomly.")
        model.eval()
        return None, model

    if version == "v9_rl_policy":
        model_name = "roberta-large"
        ckpt_path = os.path.join(ROOT, "checkpoints", "exp27_rl_policy_s3.pt")
        print(f"[S3] Loading RL Policy (RoBERTa-large) <- {os.path.basename(ckpt_path)}")
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        else:
            print(f"     [WARNING] Checkpoint not found! Using base roberta-large.")
        model.to(device)
        model.eval()
        return tok, model

    if version == "v15_t5_listwise":
        from train.exp34_train_s3_listwise import T5ListwiseScorer
        ckpt_path = os.path.join(ROOT, "checkpoints", "exp34_s3_listwise.pt")
        print(f"[S3] Loading T5 Listwise Scorer <- {os.path.basename(ckpt_path)}")
        tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = T5ListwiseScorer().to(device)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        else:
            print(f"     [WARNING] Checkpoint not found! Using untrained scorer.")
        model.eval()
        return tok, model



    if version in ["v16_bge_cross", "v17_bge_infonce"]:
        ckpt_name = "exp35_s3_cross.pt" if version == "v16_bge_cross" else "exp36_s3_infonce.pt"
        ckpt_path = os.path.join(ROOT, "checkpoints", ckpt_name)
        print(f"[S3] Loading BGE Cross-Encoder ({version}) <- {os.path.basename(ckpt_path)}")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        else:
            print(f"     [WARNING] Checkpoint not found! Using base bge-reranker-base.")
        model.eval()
        return tok, model

    if version in ["v8_gen", "v11_gen_sc", "v12_t5_mc", "v13_t5_cot", "v14_t5_pointer", "v18_t5_dpo"]:
        from transformers import T5ForConditionalGeneration
        model_name = "google/flan-t5-base"
        if version == "v18_t5_dpo":
            ckpt_path = os.path.join(ROOT, "checkpoints", "exp38_t5_dpo_s3.pt")
        elif version == "v14_t5_pointer":
            ckpt_path = os.path.join(ROOT, "checkpoints", "exp33_t5_pointer_s3.pt")
        elif version == "v13_t5_cot":
            ckpt_path = os.path.join(ROOT, "checkpoints", "exp32_t5_cot_s3.pt")
        elif version == "v12_t5_mc":
            ckpt_path = os.path.join(ROOT, "checkpoints", "exp31_t5_mc_s3.pt")
        else:
            ckpt_path = os.path.join(ROOT, "checkpoints", "exp26_t5_generative_s3.pt")
        print(f"[S3] Loading Generative Ranker (T5) <- {os.path.basename(ckpt_path)}")
        tok = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        else:
            print(f"     [WARNING] Checkpoint not found! Using base FLAN-T5-large.")
        model.to(device)
        model.eval()
        return tok, model

    if version == "v7":
        ckpt_name = "exp24_s3_path_v7.pt"
    elif version == "v6":
        ckpt_name = "exp23_s3_full_hard_neg.pt"
    elif version == "v5":
        ckpt_name = "exp19_s3_hard_negatives.pt"
    elif version == "v4":
        ckpt_name = "exp17_s3_enriched.pt"
    elif version == "v3":
        ckpt_name = "exp16v3_s3_cross.pt"
    else:
        ckpt_name = "exp16v2_s3_cross.pt"
    ckpt      = os.path.join(CKPT, ckpt_name)
    _require(ckpt)
    tok   = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"[S3] Loaded BGE-reranker-base ({version})  <- {os.path.basename(ckpt)}")
    return tok, model


# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[CDS] Checkpoint not found: {path}\n"
            "Run the corresponding training script first."
        )
