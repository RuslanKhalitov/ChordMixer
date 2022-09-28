
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



class DatasetCreator(Dataset):
    def __init__(self, df, batch_size, var_len=False):
        if var_len:
            # fill in gaps to form full batches
            df = complete_batch(df=df, batch_size=batch_size)
            # shuffle batches
            self.df = shuffle_batches(df=df)[['sequence', 'label', 'len', 'bin']]
        else:
            self.df = df
            
    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X, Y, length, bin = self.df.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.from_numpy(X)
        return (X, Y, length, bin)

    def __len__(self):
        return len(self.df)

def complete_batch(df, batch_size):
    """
    Function to make number of instances divisible by batch_size
    within each log2-bin
    """
    complete_bins = []
    bins = [bin_df for _, bin_df in df.groupby('bin')]

    for gr_id, bin in enumerate(bins):
        l = len(bin)
        remainder = l % batch_size
        integer = l // batch_size

        if remainder != 0:
            # take the first example and copy (batch_size - remainder) times
            bin = pd.concat([bin, pd.concat([bin.iloc[:1]]*(batch_size - remainder))], ignore_index=True)
            integer += 1
        batch_ids = []
        # create indices 
        for i in range(integer):
            batch_ids.extend([f'{i}_bin{gr_id}'] * batch_size)
        bin['batch_id'] = batch_ids
        complete_bins.append(bin)
    return pd.concat(complete_bins, ignore_index=True)

def shuffle_batches(df):
    """
    Shuffles batches so during training 
    ChordMixer sees sequences from different log2-bins
    """
    import random

    batch_bins = [df_new for _, df_new in df.groupby('batch_id')]
    random.shuffle(batch_bins)

    return pd.concat(batch_bins).reset_index(drop=True)


def concater_collate(batch):
    """
    Packs a batch into a long sequence
    """
    (xx, yy, lengths, bins) = zip(*batch)
    xx = torch.cat(xx, 0)
    yy = torch.tensor(yy)
    return xx, yy, list(lengths), list(bins)
