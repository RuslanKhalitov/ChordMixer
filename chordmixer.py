import math
import torch
import numpy as np
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RotateChord(nn.Module):
    """
    Parameter-free module to perform tracks shift.
    """
    def __init__(self, n_tracks, track_size):
        super(RotateChord, self).__init__()
        self.n_tracks = n_tracks
        self.track_size = track_size

    def forward(self, x, lengths=None):
        if not lengths:

            y = torch.split(
                tensor=x,
                split_size_or_sections=self.track_size,
                dim=-1
            )
            # roll sequences in a batch jointly
            z = [y[0]]
            for i in range(1, len(y)):
                offset = -2 ** (i - 1)
                z.append(torch.roll(y[i], shifts=offset, dims=1))
            z = torch.cat(z, -1)

        else:

            ys = torch.split(
                tensor=x,
                split_size_or_sections=lengths,
                dim=0
            )

            zs = []

            # roll sequences separately
            for y in ys:
                y = torch.split(
                    tensor=y,
                    split_size_or_sections=self.track_size,
                    dim=-1
                )
                z = [y[0]]
                for i in range(1, len(y)):
                    offset = -2 ** (i - 1)
                    z.append(torch.roll(y[i], shifts=offset, dims=0))
                z = torch.cat(z, -1)
                zs.append(z)
                
            z = torch.cat(zs, 0)
            assert z.shape == x.shape, 'shape mismatch'
        return z

class ChordMixerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        n_tracks,
        track_size,
        hidden_size,
        mlp_dropout,
        layer_dropout
    ):
        super(ChordMixerBlock, self).__init__()

        self.mixer = Mlp(
            embedding_size,
            hidden_size,
            embedding_size,
            act_layer=nn.GELU,
            drop=mlp_dropout
        )

        self.dropout = nn.Dropout(layer_dropout)

        self.rotator = RotateChord(n_tracks, track_size)

    def forward(self, data, lengths=None):
        res_con = data
        data = self.mixer(data)
        data = self.dropout(data)
        data = self.rotator(data, lengths)
        data = data + res_con
        return data


class ChordMixerNet(nn.Module):
    def __init__(self,
        vocab_size,
        max_seq_len,
        embedding_size,
        track_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        n_class
        ):
            super(ChordMixerNet, self).__init__()
            self.max_n_layers = math.ceil(np.log2(max_seq_len))
            n_tracks = math.ceil(np.log2(max_seq_len))
            embedding_size = int(n_tracks * track_size)
            # Init embedding layer
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_size
            )

            self.chordmixer_blocks = nn.ModuleList(
                [
                    ChordMixerBlock(
                        embedding_size,
                        n_tracks,
                        track_size,
                        hidden_size,
                        mlp_dropout,
                        layer_dropout
                    )
                    for _ in range(self.max_n_layers)
                ]
            )

            self.final =  nn.Linear(
                embedding_size,
                n_class
            )

    def forward(self, data, lengths=None):
        if lengths:
            # variable lengths mode
            n_layers = math.ceil(np.log2(lengths[0]))
        else:
            # equal lengths mode
            n_layers = self.max_n_layers
        data = self.embedding(data)
        for layer in range(n_layers):
            data = self.chordmixer_blocks[layer](data, lengths)

        # sequence-aware average pooling
        if lengths:
            data = [torch.mean(t, dim=0) for t in torch.split(data, lengths)]
            data = torch.stack(data)
        else:
            data = torch.mean(data, dim=1)
        data = self.final(data)
        return data