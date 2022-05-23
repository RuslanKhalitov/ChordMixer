import math
import torch
from torch import nn
import numpy as np
from chordmixer_block import ChordMixerBlock

class ChordMixerNet(nn.Module):
    def __init__(self,
        vocab_size,
        one_track_size,
        max_seq_len,
        mlp_cfg,
        dropout_p,
        n_class
        ):
            super(ChordMixerNet, self).__init__()
            self.vocab_size = vocab_size
            self.n_tracks = math.ceil(np.log2(max_seq_len))
            self.one_track_size = one_track_size
            self.embedding_size = int(self.n_tracks * one_track_size)
            self.max_seq_len = max_seq_len
            self.mlp_cfg = mlp_cfg #[128, 'GELU']
            self.max_n_layers = math.ceil(np.log2(max_seq_len))
            self.n_class = n_class
            self.dropout_p = dropout_p

            # Init embedding layer
            self.embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_size
            )

            self.chordmixer_blocks = nn.ModuleList(
                [
                    ChordMixerBlock(
                        vocab_size=self.vocab_size,
                        one_track_size=self.one_track_size,
                        max_seq_len=self.max_seq_len,
                        mlp_cfg=mlp_cfg,
                        dropout_p=self.dropout_p
                    )
                    for _ in range(self.max_n_layers)
                ]
            )

            self.final =  nn.Linear(
                        self.embedding_size,
                        self.n_class
                    )

    def forward(self, data):
        sequence_length = data.size(0)
        n_layers = math.ceil(np.log2(sequence_length))
        data = self.embedding(data)

        for l in range(n_layers):
            data = self.chordmixer_blocks[l](data)

        data = torch.mean(data, 0)
        data = self.final(data)
        return data
