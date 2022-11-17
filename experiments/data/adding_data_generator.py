"""
Provides the generating function for the Adding problem with variable lengths
"""
import random
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="adding_generator")
parser.add_argument("--base_length", type=str, default='200')
args = parser.parse_args()
base_length = int(args.base_length)



def sampler(base_length, min_len, max_len):
    '''
    Function to sample length values
    '''
    x = 0
    while x < min_len or x > max_len:
        rate = np.random.lognormal(0.5, 0.7, 1)
        x = int(rate * base_length)
    return x

def adding(sequences, base_length, min_len, max_len):

    sequence_list = []
    labels = []
    for _ in tqdm(range(sequences)):
        # sample length value
        n_data = sampler(base_length, min_len, max_len)
        x = np.random.uniform(-1, 1, n_data)
        y = np.zeros_like(x)

        # get relevant signal positions
        pos_1 = pos_2 = -1
        while pos_1 == pos_2:
            samples = list(random.sample(range(n_data), 2))
            samples.sort()
            pos_1, pos_2 = samples

        # fill in the relevant positions
        y[pos_1] = y[pos_2] = 1.0
        data = np.vstack([x, y]).T
        sequence_list.append(data)

        # target value
        label = 0.5 + (x[pos_1] + x[pos_2]) / 4
        labels.append(label)
        
    df = pd.DataFrame(data={'sequence': sequence_list, 'label': labels})
    return df


def generator(base_length):
    n_train = 50000 if base_length < 10**5 else 10000
    n_val = 5000 if base_length < 10**5 else 1000
    n_test = 5000 if base_length < 10**5 else 1000
    
    # to match our experiments
    boundaries = {
        200: 6700,
        1000: 31800,
        16000: 242400,
        128000: 1500000
    }
    assert base_length in boundaries.keys(), 'Incorrect base_length'
    max_possible = boundaries[base_length]

    print(f"Generating sequences for base_length = {base_length}")
    n_seq = n_train + n_val + n_test
    
    df = adding(n_seq, base_length, 32, max_possible)
    
    # calculate percentiles and split into groups
    df['len'] = df['sequence'].apply(lambda x: len(x))
    percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
    bins = np.quantile(df['len'], percentiles)
    bin_labels = [i for i in range(len(bins) - 1)]
    df['bin'] = pd.cut(df['len'], bins=bins, labels=bin_labels)
    df[['len']].to_csv(f'lenghts_{base_length}.csv', index=False)
    
    data_train, data_test = train_test_split(df, test_size=n_test+n_val, train_size=n_train)
    data_test, data_val = train_test_split(data_test, test_size=n_test, train_size=n_val)
    data_train.to_pickle(f'adding_{base_length}_train.pkl')
    data_test.to_pickle(f'adding_{base_length}_test.pkl')
    data_val.to_pickle(f'adding_{base_length}_val.pkl')

if __name__ == "__main__":
    generator(base_length=base_length)
