"""
Provides the generating function for the Adding problem with variable lengths
"""
import random
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional  as F

parser = argparse.ArgumentParser(description="adding_generator")
parser.add_argument("--base_length", type=str, default='200')
args = parser.parse_args()
base_length = int(args.base_length)

def sampler(length, min_len, max_len_observed, max_possible):
    '''
    Function to sample length values
    '''
    x = 0
    upper_bound = max_len_observed if max_len_observed != 0 else max_possible
    while x < min_len or x > upper_bound:
        rate = np.random.lognormal(0.5, 0.7, 1)
        x = int(rate * length)
        x = torch.tensor(x)
    return x

def adding(sequences, length, min_len, max_len_observed, max_possible):

    full = []
    labels = []
    lengths = []
    for _ in tqdm(range(sequences)):
        # sample length value
        n_data = sampler(length, min_len, max_len_observed, max_possible)
        lengths.append(int(n_data.numpy()))
        x = (-1 - 1) * torch.rand(n_data) + 1
        y = torch.zeros(n_data)

        # get relevant signal positions
        pos_1 = pos_2 = -1
        while pos_1 == pos_2:
            samples = list(random.sample(range(n_data), 2))
            samples.sort()
            pos_1, pos_2 = samples

        # fill the relevant positions
        y[pos_1] = y[pos_2] = 1
        data = torch.vstack([x, y]).T
        full.append(data)

        # target value
        label = 0.5 + (x[pos_1] + x[pos_2]) / 4
        labels.append(label)
    
    plt.hist(np.array(lengths), bins='auto')
    plt.title(f"Adding {length}")
    plt.savefig(f"hist_adding_{length}.jpg")

    pad_to = max(max(lengths), max_len_observed)
    new_full = []

    # padding all sequences to the maximum length
    for x in full:
        padding = (0, 0, 0, pad_to - x.shape[0])
        x = F.pad(x, padding, "constant", 0)
        new_full.append(x)
    data = torch.vstack(new_full).reshape(sequences, pad_to, 2)
    labels = torch.tensor(labels)
    return data, labels, lengths


def generator(base_length):
    n_train_seq = 50000 if base_length < 10**5 else 10000
    n_val_seq = 5000 if base_length < 10**5 else 1000
    n_test_seq = 5000 if base_length < 10**5 else 1000
    

    # from our experiments
    boundaries = {
        200: 6700,
        1000: 31800,
        16000: 242400,
        128000: 1500000
    }
    max_possible = boundaries[base_length]

    n_sequences = {
        'train': n_train_seq,
        'val': n_val_seq,
        'test': n_test_seq
    }

    print(n_sequences)
    print(f"Generating sequences for base_length = {base_length}")
    max_len_obs = 0
    for ds, n_seq in n_sequences.items():
        data, labels, lengths = adding(n_seq, base_length, 32, max_len_obs, max_possible)
        max_len_obs = max(max(lengths), max_len_obs)
        torch.save(data, f'adding_{base_length}_{ds}.pt')
        torch.save(labels, f'adding_{base_length}_{ds}_target.pt')
        df = pd.DataFrame({'lengths': lengths})
        df.to_csv(f'adding_{base_length}_lengths_{ds}.csv', index=False)


if __name__ == "__main__":
    generator(base_length=base_length)
