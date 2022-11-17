import math
import torch
import torch.nn as nn
import numpy as np
from linformer import Linformer
from nystrom_attention import Nystromformer
from reformer_pytorch import Reformer
from .poolformer import PoolFormer
from .Luna_nn import Luna
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .kernel_transformer import Kernel_transformer
from .S4_model import S4Model
from .conv_models import ConvPart
from torch.nn.utils.rnn import pack_padded_sequence
from .longformer import LongformerEncoder
import jax.numpy as jnp


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
        self.encoder = nn.Embedding(vocab_size,  dim, padding_idx=0)
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
        else:
            x = self.encoder(x)
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
        self.encoder = nn.Embedding(vocab_size, dim, padding_idx=0)
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
            x = x.permute(0, 2, 1)
            x = torch.nn.functional.pad(input=x, pad=(0, self.n_vec - x.size(2), 0, 0), mode='constant', value=0)
            x = x.permute(0, 2, 1)
            x = self.linear(x)
            x = self.linformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = torch.nn.functional.pad(input=x, pad=(0, self.n_vec - x.size(1), 0, 0), mode='constant', value=0)
            x = self.encoder(x)
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
        self.encoder = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.posenc = nn.Embedding(n_vec, dim)
        self.reformer = Reformer(
            dim=dim,
            depth=depth,
            heads=heads,
            lsh_dropout=0.1,
            causal=True
        )
        bucket_size = 64 * 2
        print('nvec', n_vec)
        self.n_vec = math.ceil(n_vec / bucket_size) * bucket_size
        print('nvec', self.n_vec)
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        # Current implementation requires equial sequence lengths 
        if self.problem == "adding":
            x = x.permute(0, 2, 1)
            x = torch.nn.functional.pad(input=x, pad=(0, self.n_vec - x.size(2), 0, 0), mode='constant', value=0)
            x = x.permute(0, 2, 1)
            x = self.linear(x)
            x = self.reformer(x)
            if self.pooling == 'avg':
                x = torch.mean(x, 1)
            elif self.pooling == 'cls':
                x = x[:, 0, :]
            x = self.final(x.view(x.size(0), -1))
        else:
            x = torch.nn.functional.pad(input=x, pad=(0, self.n_vec - x.size(1), 0, 0), mode='constant', value=0)
            x = self.encoder(x)
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
        self.encoder = nn.Embedding(vocab_size, dim, padding_idx=0)
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
        else:
            x = self.encoder(x)
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
        self.encoder = nn.Embedding(vocab_size,  dim, padding_idx=0)
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
        else:
            x = self.encoder(x).squeeze(-2)
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
        self.encoder = nn.Embedding(vocab_size,  dim, padding_idx=0)
        self.posenc = nn.Embedding(n_vec, dim)
        self.cosformer = Kernel_transformer(use_cos=True,
                                            kernel='relu',
                                            denom_eps=1e-5,
                                            d_model=dim,
                                            n_heads=heads, 
                                            n_layers=depth,
                                            n_emb=vocab_size, 
                                            problem=problem,
                                            ffn_ratio=4, 
                                            rezero=True,ln_eps=1e-5, 
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

    def forward(self, x, lengths):
        if not lengths:
            lengths = [self.n_vec]*x.shape[0]
        lengths = torch.LongTensor(lengths)
        lengths = lengths.to(self.device)
        if self.problem == "adding":
            x = self.cosformer(x, lengths=lengths)
        else:
            x = self.cosformer(x, lengths=lengths)
        if self.pooling == 'avg':
            x = torch.mean(x, 1)
        elif self.pooling == 'cls':
            x = x[:, 0, :]
        x = self.final(x.view(x.size(0), -1))
        return x

class S4_Model(nn.Module):
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
        super(S4_Model, self).__init__()
        self.device = device
        self.n_vec = int(n_vec)
        self.encoder = nn.Embedding(vocab_size,  dim, padding_idx=0)
        self.posenc = nn.Embedding(n_vec, dim)
        self.s4 = S4Model(
            d_input=dim, 
            d_output=n_class, 
            d_model=dim, 
            n_layers=depth, 
            dropout=0.)
        self.pooling = pooling
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
        else:
            x = self.encoder(x).squeeze(-2)
        x = self.s4(x)
        if self.pooling == 'avg':
            x = torch.mean(x, 1)
        elif self.pooling == 'cls':
            x = x[:, 0, :]
        x = self.final(x.view(x.size(0), -1))
        return x

class LunaModel(nn.Module):
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
        super(LunaModel, self).__init__()
        self.device = device
        self.encoder = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.posenc = nn.Embedding(n_vec, dim)
        self.luna = Luna(vocab_size, dim, depth, heads, max_length=n_vec)
        self.pooling = pooling
        self.n_vec = n_vec
        self.final = nn.Linear(dim, n_class)
        if self.pooling == 'flatten':         
            self.final = nn.Linear(dim*n_vec, n_class)
        self.problem = problem
        self.linear = nn.Linear(2, dim, bias=True)

    def forward(self, x, lengths=None):
        if self.problem == "adding":
            x = self.linear(x)
        else:
            x = self.encoder(x)
        x = self.luna(x, lengths)
        if self.pooling == 'avg':
            x = torch.mean(x, 1)
        elif self.pooling == 'cls':
            x = x[:, 0, :]
        x = self.final(x.view(x.size(0), -1))
        return x


class LSTM(nn.Module):
    def __init__(self,
        vocab_size,
        dim,
        depth,
        n_class,
        problem,
    ):
        super(LSTM, self).__init__()
        self.problem = problem
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)

        self.lstm_cell = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=depth, batch_first=True)

        self.final = nn.Linear(dim, n_class)
        self.linear = nn.Linear(2, dim)

    def forward(self, x):
        if self.problem == "adding":
            x = self.linear(x)
        else:
            x = self.embedding(x)
        # Apply LSTM
        output, (hn, cn) = self.lstm_cell(x)
        hn = hn[-1, :, :]
        
        y = self.final(hn)
        return y


class CONV(nn.Module):
    def __init__(self,
        problem,
        model,
        dim,
        depth,
        vocab_size,
        kernel_size,
        n_class
    ):
        super(CONV, self).__init__()
        self.problem = problem
        self.model = model
        self.depth  = depth
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.conv = ConvPart(model, dim, [dim] * depth, kernel_size)

        self.linear = nn.Linear(2, dim)
        self.final = nn.Linear(dim, n_class)

    def forward(self, x):

        if self.problem == "adding":
            x = self.linear(x)
        else:
            x = self.embedding(x)
        
        x = x.permute(0, 2, 1).to(dtype=torch.float)

        y_conv = self.conv(x)

        y_class = torch.mean(y_conv, dim=2)

        y = self.final(y_class)
        return y
    

class LongformerModel(nn.Module):
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
        super(LongformerModel, self).__init__()
        self.device = device
        self.n_vec = n_vec
        self.encoder = nn.Embedding(vocab_size,  dim, padding_idx=0)
        
        self.longformer_encoder = LongformerEncoder(
            vocab_size=vocab_size,
            sliding_window_size=512,
            emb_dim=n_vec,
            num_heads=heads,
            dtype=jnp.float32,
            num_layers=depth,
            qkv_dim=dim,
            mlp_dim=dim*heads,
            max_len=n_vec,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            learn_pos_emb=False,
            classifier=True,
            classifier_pool='MEAN',
            num_classes=n_class
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.longformer_encoder(x)
        return x
    