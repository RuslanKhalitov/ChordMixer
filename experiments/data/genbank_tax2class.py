import pickle5 as pickle
from genbank_taxonomy import GenBankTaxonomy
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

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

for task in tasks.keys():
    info = tasks[task]
    classes = info['classes']
    category = info['category_name']
    df = pd.read_pickle(f"{category}_df_classes.pkl")
    with open(f"{category}_taxonomies.pkl", "rb") as fh:
        data = pickle.load(fh)
    with open(f"{category}_lengths.pkl", "rb") as fh:
        lengths = pickle.load(fh)

    df['lengths'] = lengths
    df['log_lengths'] = np.log2(df['lengths'])

    df_rest = df[df['genus'].isin([classes])]
    df_rest.reset_index(inplace=True)
    df_rest = df_rest[df_rest['log_lengths'] <= 18]
    df_rest[['index', 'genus']].to_csv(f'genbank_{classes[0]}_{classes[1]}_idxs.csv', index=False)