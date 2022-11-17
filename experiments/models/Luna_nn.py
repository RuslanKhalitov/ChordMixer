# Fork from https://github.com/sooftware/luna-transformer

import math
import torch
import torch.nn as nn

from .luna_transformer.embedding import PositionalEncoding
from .luna_transformer.encoder import LunaTransformerEncoderLayer
from .luna_transformer.mask import get_attn_pad_mask


class Luna(nn.Module):
    """
    Transformer encoder architecture applied Linear Unified Nested Attention (Luna).
    Luna was proposed in the paper "Luna: Linear Unified Nested Attention" (https://arxiv.org/abs/2106.01540.pdf)
    """
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int = 6,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.1,
            project_embedding_length: int = 32,
            max_length: int = 1024
    ):
        super(Luna, self).__init__()
        self.d_model = d_model
        self.projected_embedding_length = project_embedding_length

        self.projected_embeddings = nn.Parameter(torch.Tensor(project_embedding_length, self.d_model))
        self.projected_positions = PositionalEncoding(self.d_model, project_embedding_length)
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=self.d_model ** -0.5)

        self.dropout = nn.Dropout(p=dropout_p)
        self.input_positions = PositionalEncoding(d_model, max_length)

        self.input_norm = nn.LayerNorm(d_model)
        self.embed_scale = math.sqrt(self.d_model)
        self.layers = nn.ModuleList([
            LunaTransformerEncoderLayer(
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        #print(inputs.size())
        batch_size = inputs.size()[0]
        seq_length = inputs.size()[1]

        attention_padding_mask = get_attn_pad_mask(inputs, input_lengths, self.projected_embedding_length)

        embedded = inputs

        embedded *= self.embed_scale
        projected_embedded = self.projected_embeddings * self.embed_scale

        embedded += self.input_positions(embedded.size(1))
        projected_embedded += self.projected_positions(self.projected_embedding_length).squeeze(0)

        seq_length, dim = projected_embedded.size()
        projected_embedded = projected_embedded.unsqueeze(0).expand(batch_size, seq_length, dim)

        outputs = self.dropout(embedded)
        p = self.dropout(projected_embedded)

        for layer in self.layers:
            outputs, p = layer(outputs, p, attention_padding_mask)

        return outputs