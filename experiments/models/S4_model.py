import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn

# import torchvision
# import torchvision.transforms as transforms

import os
import argparse

from .s4_src import *
from tqdm.auto import tqdm

class S4Model(nn.Module):

    def __init__(
        self, 
        d_input, 
        d_output=10, 
        d_model=256, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                    d_model=d_model, 
                    l_max=1024, 
                    bidirectional=True,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        # # Linear decoder
        # self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        #x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # # Decode the outputs
        # x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    