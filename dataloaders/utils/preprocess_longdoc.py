import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def PrepareLongdoc(data_path, max_seq_len=131000):
    """
    Adapted from 
    https://github.com/wiedersehne/Paramixer/blob/main/long-document-clf/articles_preprocessing.py
    """
    # Read the source articles
    df_cs_AI = pd.read_csv(f"{data_path}/cs.AI.csv")
    df_cs_NE = pd.read_csv(f"{data_path}/cs.NE.csv")
    df_math_AC = pd.read_csv(f"{data_path}/math.AC.csv")
    df_math_GR = pd.read_csv(f"{data_path}/math.GR.csv")

    longdoc_df = pd.concat([df_cs_AI, df_cs_NE, df_math_AC, df_math_GR], ignore_index=True)

    # Remove any arxiv tags
    longdoc_df['content'] = longdoc_df['content'].str.replace('\[\w+.\w+\]', '', regex=True)

    # Truncate to max_seq_len
    longdoc_df['len'] = longdoc_df['content'].apply(lambda x: len(x))
    longdoc_df = longdoc_df[longdoc_df['len'] <= max_seq_len]
    
    longdoc_df['label'] = longdoc_df['label'].map({'cs.AI':0, 'cs.NE':1, 'th.AC':2, 'th.GR':3})
    longdoc_df = longdoc_df[['content', 'label', 'len']]

    # Train/test split
    data_train, data_test = train_test_split(longdoc_df, test_size=0.2, stratify=longdoc_df['label'])
    data_train.to_csv(f"{data_path}/longdoc_train.csv", index=False)
    data_test.to_csv(f"{data_path}/longdoc_test.csv", index=False)


