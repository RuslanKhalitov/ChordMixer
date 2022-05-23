import math
import sys
import numpy as np
import torch
import itertools
from torch import layer_norm, nn
from typing import Union, List

import torch_geometric
from torch_sparse import spmm
from torch.utils.data import Dataset

def MakeMLP(cfg: List[Union[str, int]], in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Constructs an MLP based on a given structural config. 
    """
    layers: List[nn.Module] = []
    for i in cfg:
        if isinstance(i, int):
            layers += [nn.Linear(in_channels, i)]
            in_channels = i
        else:
            layers += [nn.GELU()]
    layers += [nn.Linear(in_channels, out_channels)]
    return nn.Sequential(*layers)
    
    
class MLPBlock(nn.Module):
    """
    Constructs a MLP with the specified structure.
    
    """
    def __init__(self, cfg, in_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.network = MakeMLP(cfg, in_dim, out_dim)

    def forward(self, data):
        return self.network(data)


class SplitAndMix(nn.Module):
    """
    Parameter free module to perform tracks shift.
    A part of the ChordMixer architecture.
    """
    def __init__(
        self,
        n_tracks,
        one_track_size
    ):
        super(SplitAndMix, self).__init__()
        self.n_tracks = n_tracks
        self.one_track_size = one_track_size
        self.embedding_size = int(n_tracks * one_track_size)

    def forward(self, data):
        '''
        Implementation for batch_size = 1
        Input = (seq_len, emb_size)
        '''

        # Split
        y = torch.split(
            tensor=data,
            split_size_or_sections=self.one_track_size,
            dim=-1
        )

        # Mix
        z = [y[0]]
        for i in range(1, len(y)):
            offset = -2 ** (i - 1)
            z.append(torch.roll(y[i], shifts=offset, dims=0))

        # Concat
        z = torch.cat(z, -1)
        return z

class ChordMixerBlock(nn.Module):
    def __init__(
        self,
        vocab_size,
        one_track_size,
        max_seq_len,
        fs,
        dropout_p,
        n_class,
        config,
        device
    ):
        super(ChordMixerBlock, self).__init__()
        self.vocab_size = vocab_size
        


class ChordMixerNet(nn.Module):
    def __init__(self,
        vocab_size,
        one_track_size,
        max_seq_len,
        fs,
        dropout_p,
        n_class,
        config,
        device
        ):
            super(ChordMixerNet, self).__init__()
            self.config = config
            self.problem = self.config['problem']
            self.vocab_size = vocab_size
            self.n_tracks = math.ceil(np.log2(max_seq_len))
            self.one_track_size = one_track_size
            self.embedding_size = int(self.n_tracks * one_track_size)
            self.max_seq_len = max_seq_len
            self.fs = fs #[128, 'GELU']
            self.max_n_layers = math.ceil(np.log2(max_seq_len))
            self.n_class = n_class
            self.dropout_p = dropout_p
            self.device = device

            # Init embedding layer
            self.embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_size
            )


            self.mlps = nn.ModuleList(
                [
                    MLPBlock(
                        self.fs,
                        self.embedding_size,
                        self.embedding_size
                    )
                    for _ in range(self.max_n_layers)
                ]
            )

            self.dropouts = nn.ModuleList(
                [
                    nn.Dropout(self.dropout_p)
                    for _ in range(self.max_n_layers)
                ]
            )
            self.final_drop = nn.Dropout(self.dropout_p)

            self.mixer = SplitAndMix(self.n_tracks, self.one_track_size)

            self.final =  nn.Linear(
                        self.embedding_size,
                        self.n_class
                    )

            self.layer_norm = nn.LayerNorm(self.embedding_size)

            self.linear_tail = nn.Linear(2, self.embedding_size)

    def forward(self, data):
        sequence_length = data.size(0)
        # this is the minimally required n_layers
        n_layers = math.ceil(np.log2(sequence_length))
        # Get embedding
        if self.problem == 'adding':
            data = self.linear_tail(data)
        else:
            data = self.embedding(data)

        for l in range(n_layers):
            # Network   
            # print(f'layer {l} before mlp:', data.size())
            res_con = data
            data = self.mlps[l](data)
            # print(f'layer {l} after mlp:', data.size())
            data = self.dropouts[l](data)
            # Mixer
            data = self.mixer(data)

            data = data + res_con
            # print(f'layer {l} after mixer:', data.size())
            # data = self.dropouts[l](data)
            # sys.exit()
        
        # print(V.shape)
        data = self.final_drop(data)
        data = torch.mean(data, 0)
        # print(f'after pooling:', data.size())
        data = self.final(data)
        # print(f'final:', data.size())
        return data
