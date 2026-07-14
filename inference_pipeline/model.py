import os, sys, json, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

class ScaledUnifiedPlanner(nn.Module):
    """
    Exp 7 Model: RoBERTa-Large based Unified Planner.
    Predicts Domain, Confidence, Relations (per-hop), and Stop signals.
    """
    def __init__(self, num_domains, num_relations, hidden_dim=512, max_hops=4):
        super().__init__()
        self.max_hops = max_hops
        
        # RoBERTa-Large Backbone
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.encoder = RobertaModel.from_pretrained("roberta-large")
        self.encoder_dim = self.encoder.config.hidden_size # 1024
        
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        
        # progressive constraint heads
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        # cross-hop reasoning heads
        self.hop_embeddings = nn.Parameter(torch.randn(max_hops, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.adaptive_stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)
        outputs = self.encoder(input_ids, attention_mask)
        
        # CLS token as global question representation
        q_h = outputs.last_hidden_state[:, 0, :] 
        h_q = self.proj(q_h) # [B, hidden_dim]
        
        # Domain and Confidence
        domain_logits = self.domain_head(h_q)
        q_confidence = torch.sigmoid(self.confidence_head(h_q))
        
        # Sequential Hop Reasoning
        # Each hop gets a refined representation via Transformer
        init_repr = h_q.unsqueeze(1) + self.hop_embeddings.unsqueeze(0) # [B, max_hops, hidden_dim]
        refined_repr = self.transformer(init_repr) # [B, max_hops, hidden_dim]
        
        rel_logits = self.relation_head(refined_repr) # [B, max_hops, num_relations]
        stop_logits = self.adaptive_stop_head(refined_repr).squeeze(-1) # [B, max_hops]
        
        return {
            'h_q': h_q,
            'refined_repr': refined_repr,
            'domain_logits': domain_logits,
            'confidence': q_confidence,
            'rel_logits': rel_logits,
            'stop_logits': stop_logits
        }
