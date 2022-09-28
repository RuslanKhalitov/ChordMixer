from training_utils import count_params, seed_everything, init_weights, train_epoch, eval_model
from dataloader_utils import DatasetCreator, concater_collate
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
import pandas as pd
import numpy as np
import wandb
import yaml 

problem_names = ['adding', 'genbank', 'longdoc']

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem_class", type=str, default='longdoc')
parser.add_argument("--problem", type=str, default='longdoc')
parser.add_argument("--model", type=str, default='chordmixer')
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--wandb", type=str, default='rusx')

args = parser.parse_args()
assert args.problem_class == 'longdoc', 'Please use the correct problem name: longdoc'

# Parsing training config
stream = open("config.yaml", 'r')
cfg_yaml = yaml.safe_load(stream)[args.problem_class]['max']
training_config = cfg_yaml['training']
print('training config', training_config)
config = cfg_yaml['models'][args.model]
print('model config', config)

model = args.model

# sys.exit()

torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)

# task variables
data_train = pd.read_pickle(f'data/longdoc_train.pkl')
data_test = pd.read_pickle(f'data/longdoc_test.pkl')
max_seq_len = max(max(data_train['len']), max(data_test['len']))

# sys.exit()
#Wandb setting
naming_log = f"{args.problem} {args.model}"
wandb.init(project="Mixer Longdoc", entity=args.wandb, name=naming_log)
wandb.config = config

if args.model == 'chordmixer':
    net = ChordMixerNet
    net = net(
        problem='longdoc',
        vocab_size=config['vocab_size'],
        max_seq_len=max_seq_len,
        embedding_size=config['embedding_size'],
        track_size = config['track_size'],
        hidden_size=config['hidden_size'],
        mlp_dropout=config['mlp_dropout'],
        layer_dropout=config['layer_dropout'],
        n_class=training_config['n_class']
    )

net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))
print(config)

class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=data_train['label'])
print('class weights:', class_weights)

loss = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float).to(device),
    reduction='mean'
)

optimizer = optim.Adam(
    net.parameters(),
    lr=config['learning_rate']
)

# Dataset preparation

# Prepare the training loader
trainset = DatasetCreator(
    df=data_train,
    batch_size=config['batch_size'],
    var_len=True
)

trainloader = DataLoader(
    trainset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=concater_collate,
    drop_last=False,
    num_workers=4
)


# Prepare the testing loader
testset = DatasetCreator(
    df=data_test,
    batch_size=config['batch_size'],
    var_len=True
)

testloader = DataLoader(
    testset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=concater_collate,
    drop_last=False,
    num_workers=4
)


for epoch in range(training_config['n_epochs']):
    print(f'Starting epoch {epoch+1}')
    train_epoch(config, net, optimizer, loss, trainloader, device=device, log_every=4000, problem='longdoc')
    test_roc_auc = eval_model(config, net, testloader, metric=accuracy_score, device=device, problem='longdoc') 
    print(f'Epoch {epoch+1} completed. Test accuracy: {test_roc_auc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{test_roc_auc:.3f}.pt')
