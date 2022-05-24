import math
import torch
import torch.nn as nn
import numpy as np
from linformer import Linformer
from nystrom_attention import Nystromformer
from reformer_pytorch import Reformer
from .poolformer import PoolFormer
# from Luna_nn import Luna
# from S4_model import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .kernel_transformer import Kernel_transformer
from .chordmixer_block import ChordMixerBlock

class ChordMixerNet(nn.Module):
    def __init__(self,
        vocab_size,
        one_track_size,
        max_seq_len,
        mlp_cfg,
        dropout_p,
        n_class,
        problem
        ):
            super(ChordMixerNet, self).__init__()
            self.problem = problem
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
            
            self.linear = nn.Linear(2, self.embedding_size, bias=True)

    def forward(self, data):
        sequence_length = data.size(0)
        n_layers = math.ceil(np.log2(sequence_length))
        if self.problem == "adding":
            data = self.linear(data)
        else:
            data = self.embedding(data)

        for l in range(n_layers):
            data = self.chordmixer_blocks[l](data)

        data = torch.mean(data, 0)
        data = self.final(data)
        return data


class TransformerModel(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem,
     pooling,
     device
     ):
        super(TransformerModel, self).__init__()
        self.device = device
        self.n_vec = n_vec
        self.encoder = nn.Embedding(vocab_size,  dim)
        self.posenc = nn.Embedding(self.n_vec, dim)
        encoder_layers = TransformerEncoderLayer(dim, heads, dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, depth)
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.transformer_encoder(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).to(self.device)
            x = self.posenc(positions) + x
            x = self.transformer_encoder(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        return x


class LinformerModel(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem,
     pooling,
     device
     ):
        super(LinformerModel, self).__init__()
        self.device = device
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.linformer = Linformer(
            dim=dim,
            seq_len=n_vec,
            depth=depth,
            heads=heads,
            k=256,
            one_kv_head=True,
            share_kv=True
        )
        self.pooling = pooling
        self.n_vec = n_vec
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':         
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.linformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).to(self.device)
            x = self.posenc(positions) + x
            x = self.linformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        return x


class ReformerModel(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem,
     pooling,
     device
     ):
        super(ReformerModel, self).__init__()
        self.device = device
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.reformer = Reformer(
            dim=dim,
            depth=depth,
            heads=heads,
            lsh_dropout=0.1,
            causal=True
        )
        self.n_vec = n_vec
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.reformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).to(self.device)
            x = self.posenc(positions) + x
            x = self.reformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        return x


class NystromformerModel(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem,
     pooling,
     device
     ):
        super(NystromformerModel, self).__init__()
        self.device = device
        self.encoder = nn.Embedding(vocab_size, dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.nystromformer = Nystromformer(
            dim=dim,
            dim_head=int(dim/heads),
            heads=heads,
            depth=depth,
            num_landmarks=256,  # number of landmarks
            pinv_iterations=6
        )
        self.n_vec = n_vec
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = self.nystromformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.encoder(x)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).to(self.device)
            #x = self.dropout1(x)
            x = self.posenc(positions) + x
            x = self.nystromformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        return x


class PoolformerModel(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem,
     pooling,
     device
     ):
        super(PoolformerModel, self).__init__()
        self.device = device
        self.n_vec = int(n_vec)
        self.encoder = nn.Embedding(vocab_size,  dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.poolformer = PoolFormer(dim, depth)
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
            x = torch.permute(x, (0,2,1))
            x = self.poolformer(x)
            x = torch.permute(x, (0,2,1))
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
        else:
            x = self.encoder(x).squeeze(-2)
            positions = torch.arange(0, self.n_vec).expand(x.size(0), self.n_vec).to(self.device)
            x = self.posenc(positions) + x
            x = torch.permute(x, (0,2,1))
            x = self.poolformer(x)
            x = torch.permute(x, (0,2,1))
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        return x


class CosformerModel(nn.Module):
    def __init__(self,
     vocab_size,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     problem,
     pooling,
     device
     ):
        super(CosformerModel, self).__init__()
        self.device = device
        self.n_vec = int(n_vec)
        self.encoder = nn.Embedding(vocab_size,  dim)
        self.posenc = nn.Embedding(n_vec, dim)
        self.cosformer = Kernel_transformer(use_cos=True,         # Whether to use the cosine reweighting mechanism prposed in the paper.
                                            kernel='relu',        # Kernel that approximates softmax. Available options are 'relu' and 'elu'.
                                            denom_eps=1e-5,       # Added to the denominator of linear attention for numerical stability.
                                            # If use_cos=True & kernel='relu' the model is equivalent to https://openreview.net/pdf?id=Bl8CQrx2Up4
                                            # If use_cos=False & kernel='elu' the model is equivalent to https://arxiv.org/pdf/2006.16236.pdf
                                            # Vanilla transformer args:
                                            d_model=dim,
                                            n_heads=heads, 
                                            n_layers=depth,
                                            n_emb=vocab_size, 
                                            problem=problem,
                                            ffn_ratio=4, 
                                            rezero=True,          # If True, use the ReZero architecture from https://arxiv.org/pdf/2003.04887.pdf, else the Pre-LN architecture from https://arxiv.org/pdf/2002.04745.pdf
                                            ln_eps=1e-5, 
                                            bias=False, 
                                            dropout=0.2, 
                                            max_len=n_vec, 
                                            xavier=True)
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.device = device

    def forward(self, x):
        lengths = [self.n_vec]*x.shape[0]
        lengths = torch.LongTensor(lengths)
        lengths = lengths.to(self.device)
        if self.problem == "adding":
            x = self.cosformer(x, lengths=lengths)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = self.cosformer(x, lengths=lengths)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        return x

