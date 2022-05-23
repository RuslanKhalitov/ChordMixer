import math
import numpy as np
import torch
from torch import nn


def MakeMLP(cfg, in_channels, out_channels):
    """
    Constructs an MLP based on a given structural config. 
    """
    layers = []
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
        mlp_cfg,
        dropout_p
    ):
        super(ChordMixerBlock, self).__init__()
        self.vocab_size = vocab_size
        self.n_tracks = self.max_n_layers = math.ceil(np.log2(max_seq_len))
        self.one_track_size = one_track_size
        self.embedding_size = int(self.n_tracks * one_track_size)
        self.mlp_cfg = mlp_cfg # e.g.: [128, 'GELU']
        self.dropout_p = dropout_p

        self.mlp = MLPBlock(
            self.mlp_cfg,
            self.embedding_size,
            self.embedding_size
        )

        self.dropout = nn.Dropout(self.dropout_p)

        self.mixer = SplitAndMix(self.n_tracks, self.one_track_size)

    def forward(self, data):
        res_con = data
        data = self.mlp(data)
        data = self.dropout(data)
        data = self.mixer(data)
        data = data + res_con
        return data

