import os
import sys
import yaml
import logging
import argparse
import torch
from models.utils import get_logger, seed_everything, count_params, init_weights
from models.backbones import ChordMixerNet, TransformerModel, LinformerModel, ReformerModel, NystromformerModel, PoolformerModel, CosformerModel

# Read arguments
parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem_class", type=str, default='adding')
parser.add_argument("--problem", type=str, default='200')
parser.add_argument("--model", type=str, default='chordmixer')
parser.add_argument("--logger", type=str, default='file')
parser.add_argument("--device_id", type=int, default=0)

args = parser.parse_args()

# Parsing training config
stream = open("config.yaml", 'r')
cfg_yaml = yaml.load(stream)[args.problem_class][args.problem]
training_config = cfg_yaml['training']
model_config = cfg_yaml['models'][args.model]

print('training config:', training_config)
print('model config:', model_config)

# Setting logger
loginf = get_logger(args, training_config, model_config)

loginf('training_config')
loginf(training_config)
loginf('model_config')
loginf(model_config)

# Fix random seeds
seed_everything(2022)

# Setting device
torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'

# Init model
model = args.model

if model == "transformer":
    net = TransformerModel
elif model == 'linformer':
    net = LinformerModel
elif model == 'reformer':
    net = ReformerModel
elif model == 'nystromformer':
    net = NystromformerModel
elif model == 'poolformer':
    net = PoolformerModel
elif model == 'cosformer':
    net = CosformerModel

if model == 'chordmixer':
    net = ChordMixerNet(
        vocab_size=model_config.vocab_size,
        one_track_size=model_config.one_track_size,
        max_seq_len=model_config.max_seq_len,
        mlp_cfg=model_config.mlp_cfg,
        dropout_p=model_config.dropout_p,
        n_class=model_config.n_class,
        problem=args.problem_class
    )
else: 
    if model == 'transformer':
        net = TransformerModel
    elif model == 'linformer':
        net = LinformerModel
    elif model == 'reformer':
        net = ReformerModel
    elif model == 'nystromformer':
        net = NystromformerModel
    elif model == 'poolformer':
        net = PoolformerModel
    elif model == 'cosformer':
        net = CosformerModel
    net = net(
        vocab_size=model_config.vocab_size,
        dim=model_config.embedding_size,
        heads=model_config.n_heads,
        depth=model_config.n_layers,
        n_vec=1000, #TODO: change it depending on the dataset
        n_class=model_config.n_class,
        pooling=model_config.pooling,
        problem=args.problem_class,
        device=device
    )

# Xavier initialization
net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))






