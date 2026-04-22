import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class QuestionEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # return CLS embedding
        return outputs.pooler_output

    @property
    def output_dim(self):
        return self.bert.config.hidden_size

class PathEncoder(nn.Module):
    def __init__(self, relation_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(relation_dim, hidden_dim, batch_first=True, bidirectional=True)
    
    def forward(self, relation_embeddings):
        # relation_embeddings: [batch, path_len, relation_dim]
        output, (hn, cn) = self.lstm(relation_embeddings)
        # concatenate the final forward and backward hidden states
        path_repr = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        return path_repr
