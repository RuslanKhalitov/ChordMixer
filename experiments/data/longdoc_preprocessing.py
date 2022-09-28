
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


config = {
    'max_seq_len': 131000,
    'vocab_size': 3574
}

def dataset_preparation(config):
    # Read the source articles
    df_cs_AI = pd.read_csv("cs.AI.csv")
    df_cs_NE = pd.read_csv("cs.NE.csv")
    df_math_AC = pd.read_csv("math.AC.csv")
    df_math_GR = pd.read_csv("math.GR.csv")

    longdoc_df = pd.concat([df_cs_AI, df_cs_NE, df_math_AC, df_math_GR], ignore_index=True)

    # Remove any arxiv tags from references
    longdoc_df['content'] = longdoc_df['content'].str.replace('\[\w+.\w+\]', '', regex=True)

    # Lengths percentiles
    longdoc_df['len'] = longdoc_df['content'].apply(lambda x: len(x))
    longdoc_df = longdoc_df[longdoc_df['len'] <= config['max_seq_len']]
    percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
    bins = np.quantile(longdoc_df['len'], percentiles)
    bin_labels = [i for i in range(len(bins) - 1)]
    longdoc_df['bin'] = pd.cut(longdoc_df['len'], bins=bins, labels=bin_labels)

    # cut documents with more than 131k symbols
    # longdoc_df['content'] = longdoc_df['content'].apply(lambda x: x[:config['max_seq_len']])
    # longdoc_df['len'] = longdoc_df['content'].apply(lambda x: len(x))
    print('max seq length:', max(longdoc_df['len']))
    
    # Create the vocabs
    texts = longdoc_df.content.values
    chars = [char for text in texts for char in text]
    chars = tuple(set(chars))
    print('dict size:', len(chars))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # Tokenize sequences
    longdoc_df['sequence'] = longdoc_df['content'].apply(lambda x: np.array([char2int[ch] for ch in x]))
    longdoc_df['label'] = longdoc_df['label'].map({'cs.AI':0, 'cs.NE':1, 'th.AC':2, 'th.GR':3})
    del(longdoc_df['content'])
    longdoc_df = longdoc_df[['sequence', 'label', 'len', 'bin']]

    # Train/val/test split
    data_train, data_test = train_test_split(longdoc_df, test_size=0.2, stratify=longdoc_df['label'])
    # data_train, data_val = train_test_split(data_train, test_size=0.2, stratify=data_train['label'])
    return data_train, data_test


data_train, data_test = dataset_preparation(
    config=config
)

data_train.to_pickle('longdoc_train_removed.pkl')
# data_val.to_pickle('longdoc_val.pkl')
data_test.to_pickle('longdoc_test_removed.pkl')


