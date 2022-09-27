import math
import torch
from torch import nn
import numpy as np
from chordmixer_block_batched import ChordMixerBlock

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

    def forward(self, data, lengths):
        n_layers = math.ceil(np.log2(lengths[0]))
        # print('n_layers', n_layers)
        data = self.embedding(data)
        # print('size after embedding', data.size())
        
        for l in range(n_layers):
            # print(f'layer {l} input:', data.size())
            data = self.chordmixer_blocks[l](data, lengths)
            # print(f'layer {l} output:', data.size())

        # print('data size before pooling', data.size())
        data = [torch.mean(t, dim=0) for t in torch.split(data, lengths)]
        data = torch.stack(data)
        # print(f'after pooling output:', data.size())
        data = self.final(data)
        # print(f'after final output:', data.size())
        return data
