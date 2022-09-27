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

    def forward(self, data, lengths):
        '''
        Implementation for batch_size = N
        Input = (
            data:     [sum(lengths), emb_size],
            lengths:  List(Int)
        )
        Output = (sum(lengths), emb_size)
        '''
        # Split on individual tensors
        # (sum(lengths), emb_size)
        # print('SM module: init size', data.size())
        ys = torch.split(
            tensor=data,
            split_size_or_sections=lengths,
            dim=0
        )
        # print('SM module: individ seq size', [i.size() for i in ys])
        # [[lengths[0], emb_size], [lengths[1], emb_size], ..., [lengths[N], emb_size]]

        # Roll each sequence individually
        zs = []
        for y in ys:

            # Split on tracks
            y = torch.split(
                tensor=y,
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
            # print('z size', z.size())
            zs.append(z)

        #Concat rolled sequences to a batch
        zs = torch.cat(zs, 0)
        # print('sizes comparison:', zs.size(), data.size())
        assert data.shape == zs.shape, "In and Out Tensors don't match"
        return zs

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

    def forward(self, data, lengths):
        res_con = data
        data = self.mlp(data)
        data = self.dropout(data)
        data = self.mixer(data, lengths)
        data = data + res_con
        return data

