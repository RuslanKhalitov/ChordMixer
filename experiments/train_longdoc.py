from training_utils import count_params, seed_everything, init_weights, train_epoch, eval_model
from dataloader_utils import DatasetCreator, concater_collate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score
from binarymixer import BinaryMixerNet
from clockwise_binarymixer import ClockwiseBinaryMixerNet
from reverse_binarymixer import ReverseBinaryMixerNet
from random_binarymixer import RandomBinaryMixerNet
from bidirectional_binarymixer import BidirectionalBinaryMixerNet
from chordmixer import ChordMixerNet

import sys
import ast
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wandb

from torch.optim.lr_scheduler import StepLR

config = {
    'model': 'bidirectional_binarymixer', #binarymixer
    'track_size': 16,
    'variable_lengths': True,
    'max_seq_len': 131000,
    'vocab_size': 4242,
    'embedding_size': 360,
    'base': 3,
    'pos_embedding': False,
    'hidden_size': 150,
    'mlp_dropout': 0.0,
    'head': 'linear', # 'linear'
    'layer_dropout': 0.2,
    'n_class': 4,
    'lr': 0.0001,   
    'n_epochs': 100,
    'batch_size': 2,
    'device_id': 0,
    'use_wandb': 1,
    'search': 0
}

naming_log = f"{config['model']}"
torch.cuda.set_device(config["device_id"])
device = 'cuda:{}'.format(config['device_id']) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)

data_train = pd.read_pickle('longdoc_train_removed.pkl')
data_test = pd.read_pickle('longdoc_test_removed.pkl')
data_val = pd.read_pickle('longdoc_val.pkl')

# print('max len train:', max(data_train['len']))
# print('max len test:', max(data_test['len']))
# print('max len val:', max(data_val['len']))

# sys.exit()

#Wandb setting
if config['use_wandb']:
    if config['search']:
        wandb.init(project="Mixer Search", entity="rusx", name=naming_log, config=config)
        config = wandb.config
    else:
        wandb.init(project="Mixer", entity="rusx", name=naming_log)
        wandb.config = config

if config['model'] == 'binarymixer':
    net = BinaryMixerNet
elif config['model'] == 'clockwise_binarymixer':
    net = ClockwiseBinaryMixerNet
elif config['model'] == 'reverse_binarymixer':
    net = ReverseBinaryMixerNet
elif config['model'] == 'random_binarymixer':
    net = RandomBinaryMixerNet
elif config['model'] == 'bidirectional_binarymixer':
    net = BidirectionalBinaryMixerNet
elif config['model'] == 'chordmixer':
    net = ChordMixerNet

net = net(
    vocab_size=config['vocab_size'],
    max_seq_len=config['max_seq_len'],
    embedding_size=config['embedding_size'],
    track_size = config['track_size'],
    hidden_size=config['hidden_size'],
    mlp_dropout=config['mlp_dropout'],
    layer_dropout=config['layer_dropout'],
    n_class=config['n_class'],
    head=config['head'],
    base=config['base']
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

optimizer = optim.AdamW(
    net.parameters(),
    lr=config['lr']
)

# scheduler = StepLR(optimizer, step_size=5, gamma=0.3)

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
    shuffle=False if config['variable_lengths'] else True,
    collate_fn=concater_collate,
    drop_last=False,
    num_workers=4
)

# Prepare the validation loader
# valset = DatasetCreator(
#     df=data_val,
#     batch_size=config['batch_size'],
#     var_len=True
# )

# valloader = DataLoader(
#     valset,
#     batch_size=config['batch_size'],
#     shuffle=False,
#     collate_fn=concater_collate,
#     drop_last=False,
#     num_workers=4
# )

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



for epoch in range(config['n_epochs']):
    print(f'Starting epoch {epoch+1}')
    train_epoch(config, net, optimizer, loss, trainloader, device=device, log_every=4000, scheduler=None)
    test_roc_auc = eval_model(config, net, testloader, metric=accuracy_score, device=device) 
    print(f'Epoch {epoch+1} completed. Test accuracy: {test_roc_auc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{test_roc_auc:.3f}.pt')


