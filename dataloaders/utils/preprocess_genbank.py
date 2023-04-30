import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

task_dict = {
    'Carassius_Labeo':
        {
            'classes': ['Carassius', 'Labeo'],
            'category_name': 'Other vertebrate',
            'max_seq_len': 100101
        },
    'Danio_Cyprinus':
        {
            'classes': ['Danio', 'Cyprinus'],
            'category_name': 'Other vertebrate',
            'max_seq_len': 261943
        },
    'Mus_Rattus':
        {
            'classes': ['Mus', 'Rattus'],
            'category_name': 'Rodent',
            'max_seq_len': 261093
        },
    'Sus_Bos':
        {
            'classes': ['Sus', 'Bos'],
            'category_name': 'Other mammalian',
            'max_seq_len': 447010
        },
}

def PrepareGenbank(data_path, taskname):
    classes = task_dict[taskname]['classes']
    max_seq_len = task_dict[taskname]['max_seq_len']
    df = pd.read_csv(f'{data_path}/{taskname}_all.csv')
    df = df[(df['len'] >= 2**5) & (df['len'] <= max_seq_len)]
    df['genus'] = df['genus'].map({classes[0]: 0, classes[1]: 1})
    df.columns = ['label', 'content', 'len']
    data_train, data_test = train_test_split(df, test_size=0.2, stratify=df['label'])
    data_train.to_csv(f"{data_path}/{taskname}_train.csv", index=False)
    data_test.to_csv(f"{data_path}/{taskname}_test.csv", index=False)


