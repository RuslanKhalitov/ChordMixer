import torch
from torch import nn
from torch.utils.data import Dataset
import logging
import os
import wandb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


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


tasks_genbank = {
    'Carassius vs. Labeo':
        {
            'classes': ['Carassius', 'Labeo'],
            'category_name': 'Other vertebrate',
            'name_raw': 'gbvrt'
        },
    'Danio vs. Cyprinus':
        {
            'classes': ['Danio', 'Cyprinus'],
            'category_name': 'Other vertebrate',
            'name_raw': 'gbvrt'
        },
    'Mus vs. Rattus':
        {
            'classes': ['Mus', 'Rattus'],
            'category_name': 'Rodent',
            'name_raw': 'gbrod'
        },
    'Sus vs. Bos':
        {
            'classes': ['Sus', 'Bos'],
            'category_name': 'Other mammalian',
            'name_raw': 'gbmam'
        },
}

gen_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7, 'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14}

def genbank_dataset_preparation(data_all, classes, pad, minimum_seq_len, maximum_seq_len, save_test_lengths):
    data_all['len'] =  data_all['sequence'].apply(lambda x: len(x))
    data_all = data_all[(data_all['len'] >= minimum_seq_len) & (data_all['len'] <= maximum_seq_len)]
    data_all = data_all[data_all['genus'].isin(classes)]
    data_all['genus'] = data_all['genus'].map({classes[0]: 0, classes[1]: 1})
    # if config['truncated_len']:
    #     data_all['sequence'] = data_all['sequence'].apply(lambda x: x[:config['truncated_len']])
    data_all['sequence'] = data_all['sequence'].apply(lambda x: [gen_dict[i] for i in x])
    max_len_observed = max(data_all['len'])
    if pad:
        def padder(one_sequence):
            l = len(one_sequence)
            n_pad = max_len_observed - l
            return np.pad(
                array=one_sequence,
                mode='constant',
                pad_width=(0, n_pad),
                constant_values=15
            )

        data_all['sequence'] = data_all['sequence'].apply(padder)

    data_train, data_test = train_test_split(data_all, test_size=0.2, stratify=data_all['genus'])
    data_train, data_val = train_test_split(data_train, test_size=0.2, stratify=data_train['genus'])
    if save_test_lengths:
        data_test[['len']].to_csv(f'genbank_{classes[0]}_{classes[1]}_test_length.csv', index=False)
    del(data_train['len'], data_test['len'],  data_val['len'])
    return data_train, data_test, data_val, max_len_observed

def adding_dataset_preparation(problem):
    data_train = torch.load(f'data/adding_{problem}_train.pt')
    labels_train = torch.load(f'data/adding_{problem}_train_target.pt')
    lengths_train_data = pd.read_csv(f'data/adding_{problem}_lengths_train.pt')
    max_len_observed = max(lengths_train_data['lengths'])

    data_val = torch.load(f'data/adding_{problem}_val.pt')
    labels_val = torch.load(f'data/adding_{problem}_val_target.pt')
    lengths_val_data = pd.read_csv(f'data/adding_{problem}_lengths_val.pt')

    data_test = torch.load(f'data/adding_{problem}_test.pt')
    labels_test = torch.load(f'data/adding_{problem}_test_target.pt')
    lengths_test_data = pd.read_csv(f'data/adding_{problem}_lengths_test.pt')
    return data_train, labels_train, lengths_train_data, \
        data_val, labels_val, lengths_val_data, \
        data_test, labels_test, lengths_test_data, max_len_observed

def longdoc_dataset_preparation():
    data_train = torch.load('data/long_document_max_train.pt').to(torch.int64)
    labels_train = torch.load('data/long_document_max_train_targets.pt').to(torch.int64)

    data_test = torch.load('data/long_document_max_test.pt').to(torch.int64)
    labels_test = torch.load('data/long_document_max_test_targets.pt').to(torch.int64)

    data_val = torch.load('data/long_document_max_val.pt').to(torch.int64)
    labels_val = torch.load('data/long_document_max_val_targets.pt').to(torch.int64)
    return data_train, labels_train, data_test, labels_test, data_val, labels_val

class GenbankDatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        _, Y, X = self.df.iloc[index, :]

        Y = torch.tensor(Y)
        X = torch.from_numpy(np.array(X))
        return (X, Y)

    def __len__(self):
        return len(self.df)


class AddingDatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data, labels, lengths, model):
        self.data = data
        self.labels = labels
        self.lengths = lengths
        self.model = model

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        if self.model == 'chordmixer':
            l = self.lengths.iloc[index, 0]
            X = X[:l]
        Y = self.labels[index]
        return (X, Y)

    def __len__(self):
        return len(self.labels)


class LongdocDatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.model = model

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        if self.model == 'chordmixer':
            # filter out PAD values
            X = X[X!=4289]
        Y = self.labels[index].to(dtype=torch.long)
        return (X, Y)

    def __len__(self):
        return len(self.labels)


def TrainModels(
    net,
    device,
    trainloader,
    valloader,
    testloader,
    optimizer,
    loss,
    args,
    model_cfg,
    loginf,
    metric,
    n_epochs,
    test_freq,
):
    problem = args.problem_class
    testing_dict = {'gt': [], 'pred': [], 'epoch': [], 'lengths': []}
    if problem == 'adding':
        lengths = pd.read_csv(f"data/adding_{args.problem}_lengths_test.csv")['lengths']
    elif problem == 'genbank':
        classes = tasks_genbank[args.problem]
        lengths =  pd.read_csv(f"data/genbank_{classes[0]}_{classes[1]}_test_length.csv")['len']
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

        t_end = datetime.now()

        msg = "Epoch {} - Training loss:  {} — Time:  {}sec".format(
            epoch,
            train_loss / len(trainloader),
            (t_end - t_start).total_seconds()
            )
        print(msg)
        loginf(msg)
        
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
                    if problem == 'adding':
                        correct_val += (torch.abs(pred.squeeze() - Y) < 0.04).sum()
                    else:
                        correct_val += predicted.eq(Y).sum().item()

                # Testing loop
                for idx, (X, Y) in enumerate(testloader):
                    X = X.to(device)
                    Y = Y.to(device)
                    pred = net(X)
                    output = loss(pred.squeeze(), Y)
                    test_loss += output.item()

                    _, predicted = pred.max(1)
                    total_test += Y.size(0)
                        
                    if problem == 'adding':
                        correct_test += (torch.abs(pred.squeeze() - Y) < 0.04).sum()
                    else:
                        correct_test += predicted.eq(Y).sum().item()


                    testing_dict['gt'].extend(Y.detach().cpu().numpy())
                    testing_dict['pred'].extend(predicted.detach().cpu().numpy())
                    testing_dict['epoch'].extend([epoch for _ in range(Y.size(0))])

            lengths_epoch = lengths[:(len(testloader) * model_cfg['batch_size'])]
            testing_dict['lengths'].extend(lengths_epoch)
            testing_df = pd.DataFrame(testing_dict)
            testing_df.to_csv(f'{problem}_testing_df.csv', index=False)

            msg = "Val loss: {}".format(val_loss / len(valloader))
            print(msg)
            loginf(msg)
            msg = "Test loss: {}".format(test_loss / len(testloader))
            print(msg)
            loginf(msg)
            
            accuracy_val = 100.*correct_val/total_val
            accuracy_test = 100.*correct_test/total_test
            msg = "Val  accuracy: {}".format(accuracy_val)
            print(msg)
            loginf(msg)
            msg = "Test accuracy: {}".format(accuracy_test)
            print(msg)
            loginf(msg)

            loginf({"test_loss": test_loss / len(valloader)})
            loginf({"val_loss": val_loss / len(valloader)})
            loginf({"val_acc": accuracy_val})
            loginf({"test_acc": accuracy_test})

            if metric == 'rocauc':
                epoch_df_test = testing_df[testing_df['epoch'] == epoch]
                roc_test = 100.*roc_auc_score(list(epoch_df_test['gt']), list(epoch_df_test['pred']))
                loginf({"test_roc_auc": roc_test})

            print('_' * 40)
            net.train()
            