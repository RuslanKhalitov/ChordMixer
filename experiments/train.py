import os
import sys
import yaml
import logging
import argparse
import torch
from torch import nn, optim
import pandas as pd
from models.utils import get_logger, seed_everything, count_params, \
    init_weights, genbank_dataset_preparation, adding_dataset_preparation, \
    longdoc_dataset_preparation, tasks_genbank, TrainModels, \
    GenbankDatasetCreator, AddingDatasetCreator, LongdocDatasetCreator

from sklearn.utils.class_weight import compute_class_weight

from models.backbones import ChordMixerNet, TransformerModel, LinformerModel, \
     ReformerModel, NystromformerModel, PoolformerModel, CosformerModel, \
     LunaModel, S4_Model

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
model = args.model

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

# Reading the data and setting the dataloaders
if args.problem_class == 'genbank':
    classes = tasks_genbank[args.problem]['classes']
    data_all = pd.read_csv(f'data/genbank_{classes[0]}_{classes[1]}_data.csv')
    data_train, data_test, data_val, max_length_obs = genbank_dataset_preparation(
        pad=False if model=='chordmixer' else True,
        data_all=data_all,
        classes=classes,
        minimum_seq_len=2**5,
        maximum_seq_len=2**19,
        save_test_lengths=True
    )
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=data_train['genus'])


    trainset = GenbankDatasetCreator(
        df=data_train,
        trunc=model_config['truncation']
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=model_config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    valset = GenbankDatasetCreator(
        df=data_val,
        trunc=model_config['truncation']
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    testset = GenbankDatasetCreator(
        df=data_test,
        trunc=model_config['truncation']
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4
    )
elif args.problem_class == 'adding':
    data_train, labels_train, lengths_train_data, \
        data_val, labels_val, lengths_val_data, \
        data_test, labels_test, lengths_test_data, max_len_observed = adding_dataset_preparation(
            problem=args.problem
        )
    
    trainset = AddingDatasetCreator(
        data=data_train,
        labels=labels_train,
        lengths=lengths_train_data,
        model=model
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=model_config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # Prepare the validation loader
    valset = AddingDatasetCreator(
        data=data_val,
        labels=labels_val,
        lengths=lengths_val_data,
        model=model
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    # Prepare the testing loader
    testset = AddingDatasetCreator(
        data=data_test,
        labels=labels_test,
        lengths=lengths_test_data,
        model=model
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4
    )
elif args.problem_class == 'longdoc':
    data_train, labels_train, data_test,\
    labels_test, data_val, labels_val = longdoc_dataset_preparation()
    max_len_observed = 131000
    class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=list(labels_train))

    trainset = LongdocDatasetCreator(
        data=data_train,
        labels=labels_train,
        model=model
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=model_config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # Prepare the validation loader
    valset = LongdocDatasetCreator(
        data=data_val,
        labels=labels_val,
        model=model
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    # Prepare the testing loader
    testset = LongdocDatasetCreator(
        data=data_test,
        labels=labels_test,
        model=model
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

# Init the model

if model == 'chordmixer':
    net = ChordMixerNet(
        vocab_size=model_config.vocab_size,
        one_track_size=model_config.one_track_size,
        max_seq_len=max_length_obs,
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
    elif model == 'luna':
        net = LunaModel
    elif model == 's4':
        net = S4_Model
    net = net(
        vocab_size=model_config.vocab_size,
        dim=model_config.embedding_size,
        heads=model_config.n_heads,
        depth=model_config.n_layers,
        n_vec=model_config['truncation'] if model_config['truncation'] else max_length_obs,
        n_class=model_config.n_class,
        pooling=model_config.pooling,
        problem=args.problem_class,
        device=device
    )

# Xavier initialization
net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))

# Loss and optimizer
if args.problem_class == 'adding':
    loss = nn.MSELoss()
else: 
    loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device),
        reduction='mean'
    )
optimizer = optim.Adam(
        net.parameters(),
        lr=model_config['learning_rate']
    )


TrainModels(
    net,
    device,
    trainloader,
    valloader,
    testloader,
    optimizer,
    loss,
    args,
    loginf,
    model_cfg=model_config,
    metric='accuracy' if args.problem_class == 'adding' else 'rocauc',
    n_epochs=training_config['n_epochs'],
    test_freq=1,
)






