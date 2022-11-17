from training_utils import count_params, seed_everything, init_weights, train_epoch, eval_model
from dataloader_utils import DatasetCreator, concater_collate, DatasetCreatorFlat
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score
from chordmixer import ChordMixerNet

import argparse
import sys
import ast
import math
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR
import pandas as pd
import numpy as np
import wandb
import yaml 

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem", type=str, default='cifar10')
parser.add_argument("--model", type=str, default='chordmixer')
parser.add_argument("--device_id", type=int, default=2)
parser.add_argument("--wandb", type=str, default='rusx')

args = parser.parse_args()

config = {
    'max_seq_len': 1024,
    'search': False,
    'vocab_size': 256,
    'track_size': 30,
    'embedding_size': 240,
    'hidden_size': 196,
    'mlp_dropout': 0.,
    'layer_dropout': 0.,
    'n_class': 10,
    'n_epochs': 150,
    'positional_embedding': False,
    'permutation': False,
    'embedding_type': 'linear',
    'learning_rate': 0.001,
    'batch_size': 96
}


torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)

# task variables
data_train = torch.load('cifar10_train.pt').to(torch.int32)
data_train = data_train.to(torch.int32) if config['embedding_type'] == 'sparse' else data_train.to(torch.float)
labels_train = torch.load('cifar10_labels_train.pt').to(torch.int32)

data_test = torch.load('cifar10_test.pt')
data_test = data_test.to(torch.int32) if config['embedding_type'] == 'sparse' else data_test.to(torch.float)
labels_test = torch.load('cifar10_labels_test.pt').to(torch.int32)

naming_log = "CIFAR10"
if not config['search']:
    wandb.init(project="ChordMixer LRA", entity=args.wandb, name=naming_log)
    wandb.config = config
else:
    wandb.init(project="ChordMixer Sweep", entity=args.wandb, config=config)
    config = wandb.config
    print('CONFIG')
    print(config)
    

net = ChordMixerNet(
    vocab_size=config['vocab_size'],
    max_seq_len=config['max_seq_len'],
    embedding_size=config['embedding_size'],
    track_size = config['track_size'],
    hidden_size=config['hidden_size'],
    mlp_dropout=config['mlp_dropout'],
    layer_dropout=config['layer_dropout'],
    n_class=config['n_class'],
    positional_embedding=config['positional_embedding'],
    embedding_type=config['embedding_type'],
)

net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))
print(config)

loss = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    net.parameters(),
    lr=config['learning_rate'],
    betas = (0.9, 0.98), eps = 1e-8, weight_decay=0.01
)

scheduler = LinearLR(optimizer, start_factor=1e-7, total_iters=100)

# Prepare the training loader
trainset = DatasetCreatorFlat(
    df=data_train,
    labels=labels_train,
    perm=None
)

trainloader = DataLoader(
    trainset,
    batch_size=config['batch_size'],
    shuffle=True,
    drop_last=False,
    num_workers=4
)


# Prepare the testing loader
testset = DatasetCreatorFlat(
    df=data_test,
    labels=labels_test,
    perm=None
)

testloader = DataLoader(
    testset,
    batch_size=config['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=4
)

for epoch in range(config['n_epochs']):

    print(f'Starting epoch {epoch+1}')
    train_epoch(config, net, optimizer, loss, trainloader, device=device, scheduler=scheduler, log_every=10000)
    acc = eval_model(config, net, testloader, metric=accuracy_score, device=device) 
    print(f'Epoch {epoch+1} completed. Test accuracy: {acc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{acc:.3f}.pt')

