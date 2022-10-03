
import os
import sys
import math
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import wandb

def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return n_params

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_epoch(config, net, optimizer, loss, trainloader, device, log_every=50, scheduler=None, problem=None):
    net.train()
    running_loss = 0.0
    n_items_processed = 0
    num_batches = len(trainloader)
    for idx, (X, Y, length, bin) in tqdm(enumerate(trainloader), total=num_batches):
        if problem == 'adding':
            X = X.float().to(device)
            Y = Y.float().to(device)
            output = net(X, length).squeeze()
            output = loss(output, Y)
        else:
            X = X.to(device)
            Y = Y.to(device)            
            output = net(X, length)
            output = loss(output, Y)

        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        running_loss += output.item()
        n_items_processed += len(length)

        if (idx + 1) % log_every == 0:
            print(f'Avg loss after {idx + 1} batches: {running_loss / n_items_processed}')
            # print(f'Current loss on a batch {idx + 1}: {output.item() / len(length)}')

    total_loss = running_loss / num_batches
    print(f'Training loss after epoch: {total_loss}')
    wandb.log({'train loss': total_loss})

def eval_model(config, net, valloader, metric, device, problem) -> float:
    net.eval()

    preds = []
    targets = []
    bins = []

    num_batches = len(valloader)
    for idx, (X, Y, length, bin) in tqdm(enumerate(valloader), total=num_batches, position=0, leave=True, ascii=False):
        if problem == 'adding':
            X = X.float().to(device)
            Y = Y.float().to(device)
            output = net(X, length).squeeze()
            predicted = output
        else:
            X = X.to(device)
            Y = Y.to(device)  
            output = net(X, length) 
            _, predicted = output.max(1)
        
        targets.extend(Y.detach().cpu().numpy().flatten())
        preds.extend(predicted.detach().cpu().numpy().flatten())
        bins.extend(bin)

    total_metric = metric(preds, targets)
    
    results = pd.DataFrame(data={'bins': bins, 'predictions': preds, 'labels': targets})

    # Calculate scores for each percentile
    def percentile_scores(df):
        scores_dict = {}
        for i in sorted(df['bins'].unique()):
            data = df[df['bins'] == i]
            try:
                scores_dict[f'test_bin_{i}'] = metric(data['predictions'], data['labels'])
            except:
                "Can't calculate ROCAUC because only one class appears in the percentile"
                scores_dict[f'test_bin_{i}'] = 0.5
        wandb.log(scores_dict)


    percentile_scores(results)
    wandb.log({'test metric': total_metric})

    return total_metric


def inference(config, net, testloader, device) -> pd.DataFrame:
    net.eval()

    preds = []
    num_batches = len(testloader)

    for idx, (X, Y, length, bin) in tqdm(enumerate(testloader), total=num_batches, position=0, leave=True, ascii=False):
        X = X.to(device)
        output = net(X, length)
        preds.extend(output.detach().cpu().numpy().flatten())
        
    return pd.DataFrame({
        'prediction': preds
    })