"""
exp29_s2_gnn_ranker.py
======================

Defines the GraphAwareRanker architecture. It enhances the Stage 2 MPNet ranker
by incorporating lightweight structural graph features (e.g., node degree, path length)
directly into the MLP fusion head.

Usage:
  Imported by train/ and cds_pipeline/models.py
"""

import torch
import torch.nn as nn
from transformers import AutoModel

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
