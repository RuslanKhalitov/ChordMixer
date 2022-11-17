"""
Implementation of TCN and 1-D CCNN models
Fork from 
https://github.com/locuslab/TCN
"""

import sys
import torch
import torch.nn as nn
import torchvision.ops
from torch.nn.utils import weight_norm


# TCN
class tcn(nn.Module):
    def __init__(self, tcn_size):
        super(tcn, self).__init__()
        self.tcn_size = tcn_size

    def forward(self, x):
        x_new = x[:, :, :-self.tcn_size]
        return x_new.contiguous()



# One Conv. block
class Block(nn.Module):
    def __init__(self, model, c_in, c_out, ks, pad, dil):
        super(Block, self).__init__()
        self.model = model

        if model == 'CDIL':
            pad_mode = 'circular'
        else:
            pad_mode = 'zeros'

        self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode=pad_mode))
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.normal_(0, 0.01)

        if model == 'TCN':
            self.cut = tcn(pad)
            self.tcn = nn.Sequential(self.conv, self.cut)

        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        self.nonlinear = nn.ReLU()

    def forward(self, x):
        if self.model == 'TCN':
            net = self.tcn
        else:
            net = self.conv

        out = net(x)
        res = x if self.res is None else self.res(x)
        return self.nonlinear(out) + res


# Conv. blocks
class ConvPart(nn.Module):
    def __init__(self, model, dim_in, hidden_channels, ks):
        super(ConvPart, self).__init__()
        layers = []
        num_layer = len(hidden_channels)
        begin = 1
        for i in range(begin, num_layer):
            this_in = dim_in if i == 0 else hidden_channels[i - 1]
            this_out = hidden_channels[i]
            if model == 'CNN':
                this_dilation = 1
                this_padding = int((ks - 1) / 2)
            else:
                this_dilation = 2 ** i
                if model == 'TCN':
                    this_padding = this_dilation * (ks - 1)
                elif model == 'CDIL' or model == 'DIL':
                    this_padding = int(this_dilation*(ks-1)/2)
                else:
                    print('unknown model.')
                    sys.exit()
            if i < (num_layer-3):
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation)]
            else:
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)
