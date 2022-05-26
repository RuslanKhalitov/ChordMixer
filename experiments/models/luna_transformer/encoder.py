# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

from .attention import LinearUnifiedNestedAttention
from .feed_forward import PositionwiseFeedForwardNetwork


class LunaTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.3,
    ) -> None:
        super(LunaTransformerEncoderLayer, self).__init__()
        self.luna_attention = LinearUnifiedNestedAttention(d_model, num_attention_heads)
        self.feed_forward = PositionwiseFeedForwardNetwork(d_model, d_ff, dropout_p)
        self.packed_context_layer_norm = nn.LayerNorm(d_model)
        self.unpacked_context_layer_norm = nn.LayerNorm(d_model)
        self.unpacked_context_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(
            self,
            inputs: torch.FloatTensor,
            p: torch.FloatTensor,
            attention_padding_mask: torch.FloatTensor = None,
    ):
        unpacked_context, packed_context = self.luna_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            p=p,
            attention_padding_mask=attention_padding_mask,
        )

        packed_context = self.packed_context_layer_norm(packed_context + p)
        unpacked_context = self.unpacked_context_layer_norm(unpacked_context + inputs)

        outputs = self.feed_forward(unpacked_context)
        outputs = self.feed_forward_layer_norm(outputs + unpacked_context)

        return outputs, packed_context
