"""Official PyTorch-only research inference. No substitute or fallback runtime."""
import torch
from torch import nn


class MiniFormulaLM(nn.Module):
    def __init__(self, payload):
        super().__init__()
        self.labels = payload['labels']
        self.relations = payload['relations']
        self.max_positions = payload['max_positions']
        hidden = payload['hidden']
        self.pad_id = len(self.labels) + len(self.relations)
        self.cls_id, self.sep_id, self.mask_id = self.pad_id+1, self.pad_id+2, self.pad_id+3
        self.token_embedding = nn.Embedding(self.pad_id+4, hidden, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(self.max_positions, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=payload['heads'],
                    dim_feedforward=payload['feedforward'], dropout=.1,
                    activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=payload['layers'], enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, len(self.labels))

    def forward(self, input_ids, attention, mask_positions):
        if input_ids.dtype != torch.long or attention.dtype != torch.bool or mask_positions.dtype != torch.long:
            raise ValueError('invalid request dtype')
        if input_ids.ndim != 2 or attention.shape != input_ids.shape or mask_positions.shape != (len(input_ids),):
            raise ValueError('invalid request shape')
        if input_ids.shape[1] > self.max_positions or not attention.any(dim=1).all():
            raise ValueError('invalid sequence length or empty attention')
        if (input_ids < 0).any() or (input_ids >= self.token_embedding.num_embeddings).any():
            raise ValueError('request vocabulary outside checkpoint')
        if (mask_positions < 0).any() or (mask_positions >= input_ids.shape[1]).any():
            raise ValueError('target outside sequence')
        batch = torch.arange(len(input_ids),device=input_ids.device)
        if not (input_ids[batch,mask_positions] == self.mask_id).all() or not attention[batch,mask_positions].all():
            raise ValueError('target must be attended mask token')
        positions=torch.arange(input_ids.shape[1],device=input_ids.device).unsqueeze(0)
        hidden=self.token_embedding(input_ids)+self.position_embedding(positions)
        encoded=self.encoder(hidden,src_key_padding_mask=~attention)
        return self.classifier(self.output_norm(encoded[batch,mask_positions]))


def load_model(checkpoint):
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['schema'] != 'aiflow-prompt-mini-lm-distillation/v1':
        raise ValueError('unsupported checkpoint schema')
    model=MiniFormulaLM(payload)
    model.load_state_dict(payload['state_dict'],strict=True)
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('checkpoint requires finite FP32 weights')
    return model.eval(),payload
