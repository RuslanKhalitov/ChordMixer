import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

tasks = {
    'Carassius vs. Labeo':
        {
            'classes': ['Carassius', 'Labeo'],
            'category_name': 'Other vertebrate',
            'name_raw': 'gbvrt',
            'max_seq_len': 100101
        },
    'Danio vs. Cyprinus':
        {
            'classes': ['Danio', 'Cyprinus'],
            'category_name': 'Other vertebrate',
            'name_raw': 'gbvrt',
            'max_seq_len': 261943
        },
    'Mus vs. Rattus':
        {
            'classes': ['Mus', 'Rattus'],
            'category_name': 'Rodent',
            'name_raw': 'gbrod',
            'max_seq_len': 261093
        },
    'Sus vs. Bos':
        {
            'classes': ['Sus', 'Bos'],
            'category_name': 'Other mammalian',
            'name_raw': 'gbmam',
            'max_seq_len': 447010
        },
}


gen_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7, 'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14}

def dataset_preparation(classes, minimum_seq_len, maximum_seq_len):

    data_all = pd.read_csv(f'genbank_{classes[0]}_{classes[1]}_data.csv')

    data_all = data_all[data_all['genus'].isin(classes)]
    data_all['genus'] = data_all['genus'].map({classes[0]: 0, classes[1]: 1})

    data_all['len'] =  data_all['sequence'].apply(lambda x: len(x))
    data_all = data_all[(data_all['len'] >= minimum_seq_len) & (data_all['len'] <= maximum_seq_len)]

    percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
    bins = np.quantile(data_all['len'], percentiles)
    bin_labels = [i for i in range(len(bins) - 1)]
    data_all['bin'] = pd.cut(data_all['len'], bins=bins, labels=bin_labels)

    data_all['sequence'] = data_all['sequence'].apply(lambda x: np.array([gen_dict[i] for i in x]))
    data_all = data_all[['sequence', 'genus', 'len', 'bin']]
    data_all.columns = ['sequence', 'label', 'len', 'bin']
    data_train, data_test = train_test_split(data_all, test_size=0.2, stratify=data_all['label'])
    data_train, data_val = train_test_split(data_train, test_size=0.2, stratify=data_train['label'])

    return data_train, data_test, data_val


if __name__ == "__main__":
    for task in tasks.keys():
        print(task)
        data = tasks[task]
        class_1 = data['classes'][0]
        class_2 = data['classes'][1]
        data_train, data_test, data_val = dataset_preparation(
            classes=data['classes'],
            minimum_seq_len=2**5,
            maximum_seq_len=data['max_seq_len']
        )

        data_train.to_pickle(f'{class_1}_{class_2}_train.pkl')
        data_test.to_pickle(f'{class_1}_{class_2}_test.pkl')