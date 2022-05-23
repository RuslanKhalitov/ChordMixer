import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
        

def build_dataset():
    """
    70% train 20% val and 10% test
    :return:
    """
    df_cs_AI = pd.read_csv("cs.AI.csv")
    df_cs_NE = pd.read_csv("cs.NE.csv")
    df_math_AC = pd.read_csv("math.AC.csv")
    df_math_GR = pd.read_csv("math.GR.csv")

    df_cs_AI_train, df_cs_AI_val, df_cs_AI_test = np.split(df_cs_AI,
                                                           [int(.7*len(df_cs_AI)), int(.9*len(df_cs_AI))])
    df_cs_NE_train, df_cs_NE_val, df_cs_NE_test = np.split(df_cs_NE,
                                                           [int(.7 * len(df_cs_NE)), int(.9 * len(df_cs_NE))])
    df_math_AC_train, df_math_AC_val, df_math_AC_test = np.split(df_math_AC,
                                                           [int(.7 * len(df_math_AC)), int(.9 * len(df_math_AC))])
    df_math_GR_train, df_math_GR_val, df_math_GR_test = np.split(df_math_GR,
                                                                 [int(.7 * len(df_math_GR)), int(.9 * len(df_math_GR))])
    #print(df_cs_AI_train.head(5), df_cs_AI_val.shape, df_cs_AI_test.shape)

    frames = [df_cs_AI_train, df_cs_NE_train, df_math_GR_train, df_math_AC_train]
    train_df = pd.concat(frames)
    frames = [df_cs_AI_val, df_cs_NE_val, df_math_GR_val, df_math_AC_val]
    val_df = pd.concat(frames)
    frames = [df_cs_AI_test, df_cs_NE_test, df_math_GR_test, df_math_AC_test]
    test_df = pd.concat(frames)

    print(train_df.shape, val_df.shape, test_df.shape)

    train_df.to_csv("train_ld.csv", index=False)
    val_df.to_csv("val_ld.csv", index=False)
    test_df.to_csv("test_ld.csv", index=False)
    
def label(x):
    if x == "cs.AI":
        return 0
    elif x == "cs.NE":
        return 1
    elif x == "th.AC":
        return 2
    elif x == "th.GR":
        return 3

def read_articles():
    train_df = pd.read_csv('train_ld.csv')
    val_df = pd.read_csv('val_ld.csv')
    test_df = pd.read_csv('test_ld.csv')
    # Convert sentiment columns to numerical values
    train_df.label = train_df.label.apply(lambda x: label(x))
    val_df.label = val_df.label.apply(lambda x: label(x))
    test_df.label = test_df.label.apply(lambda x: label(x))
    print(train_df.shape)
    return train_df, val_df, test_df

def get_char_index(train_df, val_df, test_df):
    """
    get content and size of vocabulary
    """
    df = pd.concat([train_df, val_df, test_df])
    texts = df.content.values
    chars = [char for text in texts for char in text]
    chars = tuple(set(chars))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    vocab_size = len(chars)+1
    char2int['<PAD>'] = vocab_size - 1
    print('Pad token for vocab is: ', char2int['<PAD>'])
    return char2int

def text_to_sequence(texts, token_index):
    """
    transform text to vector
    :param texts:
    :param token_index: vocabulary: {8, 'Y': 9, 'z': 10, ' ': 11}
    :return: encoded sequence [8, 12, 34, 56, 14]
    """
    encoded_sequences = []
    for text in texts:
        encodes = np.array([token_index[ch] for ch in text])
        encoded_sequences.append(encodes)
    return encoded_sequences

def get_articles_data(max_len):
    """
    save articles as tensors for training
    """
    # STEP 1: read data and get vocabulary
    train_df, val_df, test_df = read_articles()
    token_index = get_char_index(train_df, val_df, test_df)
    print(len(token_index))

    # STEP 2: transform text to sequence with the vocab.
    X_train = text_to_sequence(train_df.content.values, token_index)
    X_val = text_to_sequence(val_df.content.values, token_index)
    X_test = text_to_sequence(test_df.content.values, token_index)

    sequences = [len(s) for s in X_train]
    print(f"average training sequence length {np.mean(sequences)}")

    y_train = train_df["label"]
    y_test = test_df["label"]
    y_val = val_df["label"]

    # STEP 3: zero padding
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len, value=token_index['<PAD>'],
                                                            padding="post", truncating="post")
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len, value=token_index['<PAD>'],
                                                           padding="post", truncating="post")
    X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_len, value=token_index['<PAD>'],
                                                           padding="post", truncating="post")

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    X_val = torch.tensor(X_val)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    y_val = torch.tensor(y_val)

    print(X_train.shape, X_test.shape, X_val.shape)

    torch.save(X_train, 'long_document_max_train.pt')
    torch.save(y_train, 'long_document_max_train_targets.pt')

    torch.save(X_test, 'long_document_max_test.pt')
    torch.save(y_test, 'long_document_max_test_targets.pt')

    torch.save(X_val, 'long_document_max_val.pt')
    torch.save(y_val, 'long_document_max_val_targets.pt')

    return X_train, y_train, X_test, y_test, X_val, y_val

if __name__ == "__main__":
    build_dataset()
    get_articles_data(131000)