import torch
import torch.nn as nn
from transformers import AutoModel

class ViMedEmbeddingModel(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", embedding_dim=768):
        super().__init__()
        self.HF_TOKEN = "*********************" #fill your's token in hugging face
        self.backbone = AutoModel.from_pretrained(model_name, token = self.HF_TOKEN)
        self.projection = nn.Linear(self.backbone.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        emb = self.projection(cls_emb)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb
