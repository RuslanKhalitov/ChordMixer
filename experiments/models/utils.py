import torch
from torch import nn
import logging
import os
import wandb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score


def init_weights(m):
    """
    Xavier weights initialization for linear layers
    :param m: model parameters
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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

def get_logger(args):
    if args.logger == 'file':
        filename = f'{args.problem_class}_{args.problem}_{args.model}.txt'
        os.makedirs("logfiles", exist_ok=True)
        log_file_name = "./logfiles/" + filename
        handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
        loginf = logging.info
        loginf(log_file_name)
        return loginf

    elif args.logger == 'wandb':
        pass


def TrainModels(
        net,
        device,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        test_freq,
        optimizer,
        loss,
        metric,
        segment_size,
        config
):
    testing_dict = {'gt': [], 'pred': [], 'epoch': [], 'lengths': []}
    if config['problem'] == 'adding':
        lengths = pd.read_csv(f"../../../../data/shuttlenet/genbank/adding_lengths_{config['how_long']}.csv")['lengths']
    else:
        lengths =  pd.read_csv(f"../../../../data/shuttlenet/genbank/test_length_{config['naming']}.csv")['len']
    for epoch in range(n_epochs):
        # Training
        train_loss = 0.
        t_start = datetime.now()
        for idx, (X, Y) in tqdm(enumerate(trainloader), total=len(trainloader)):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            pred = net(X)
            output = loss(pred.squeeze(), Y)
            output.backward()
            train_loss += output.item()

            optimizer.step()
            _, predicted = pred.max(1)

        t_end = datetime.now()

        print("Epoch {} - Training loss:  {} — Time:  {}sec".format(
            epoch,
            train_loss / len(trainloader),
            (t_end - t_start).total_seconds()
            )
        )
        
        # Validation
        if epoch % test_freq == 0:
            net.eval()
            total_val = 0
            total_test = 0
            correct_val = 0
            correct_test = 0
            val_loss = 0.
            test_loss = 0.
            with torch.no_grad():
                # Validation loop
                for _, (X, Y) in enumerate(valloader):
                    X = X.to(device)
                    Y = Y.to(device)
                    pred = net(X)
                    output = loss(pred.squeeze(), Y)
                    val_loss += output.item()
                    _, predicted = pred.max(1)
                    total_val += Y.size(0)
                    if config['problem'] == 'genbank':
                        correct_val += predicted.eq(Y).sum().item()
                    elif config['problem'] == 'adding':
                        correct_val += (torch.abs(pred.squeeze() - Y) < 0.04).sum()
                        predicted = pred.squeeze()

                # Testing loop
                for idx, (X, Y) in enumerate(testloader):
                    X = X.to(device)
                    Y = Y.to(device)
                    pred = net(X)
                    output = loss(pred.squeeze(), Y)
                    test_loss += output.item()

                    _, predicted = pred.max(1)
                    total_test += Y.size(0)
                    if config['problem'] == 'genbank':
                        correct_test += predicted.eq(Y).sum().item()
                    elif config['problem'] == 'adding':
                        correct_test += (torch.abs(pred.squeeze() - Y) < 0.04).sum()
                        predicted = pred.squeeze()


                    testing_dict['gt'].extend(Y.detach().cpu().numpy())
                    testing_dict['pred'].extend(predicted.detach().cpu().numpy())
                    testing_dict['epoch'].extend([epoch for _ in range(Y.size(0))])
            lengths_epoch = lengths[:(len(testloader) * config['batch_size'])]
            testing_dict['lengths'].extend(lengths_epoch)
            testing_df = pd.DataFrame(testing_dict)
            testing_df.to_csv(f'{config["naming"]}_testing_df.csv', index=False)

            print("Val  loss: {}".format(val_loss / len(valloader)))
            print("Test loss: {}".format(test_loss / len(testloader)))
            accuracy_val = 100.*correct_val/total_val
            accuracy_test = 100.*correct_test/total_test
            print("Val  accuracy: {}".format(accuracy_val))
            print("Test accuracy: {}".format(accuracy_test))
            if config['use_wandb']:
                wandb.log({"test_loss": test_loss / len(valloader)})
                wandb.log({"val_loss": val_loss / len(valloader)})
                wandb.log({"val_acc": accuracy_val})
                wandb.log({"test_acc": accuracy_test})

            
            if metric == 'rocauc':
                epoch_df_test = testing_df[testing_df['epoch'] == epoch]
                roc_test = 100.*roc_auc_score(list(epoch_df_test['gt']), list(epoch_df_test['pred']))
                if config['use_wandb']:
                    wandb.log({"test_roc_auc": roc_test})

            print('_' * 40)
            net.train()
            