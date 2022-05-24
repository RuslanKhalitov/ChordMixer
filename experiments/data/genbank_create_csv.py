import glob
import gzip
import pickle5 as pickle
from Bio import GenBank
import pandas as pd
from tqdm import tqdm

tasks = {
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


def parse_category_special(df, path_to):
    """
    Iteratively parses all raw files of the category 
    and saves sequences with the relevant indices
    """
    files = glob.glob(f"{path_to}/*.seq.gz")
    n_file = len(files)
    seqs = []
    needed_idx = list(df['index'])
    idx = 0
    for fi in tqdm(range(n_file)):
        fname = f"{path_to}{name_raw}{fi + 1}.seq.gz"
        with gzip.open(fname, "rt") as handle:
            for record in GenBank.parse(handle):
                if idx in needed_idx:
                    seqs.append(record.sequence)
                idx += 1
    df['sequence'] = seqs
    return df

for task in tasks.keys():
    info = tasks[task]
    classes = info['classes']
    category = info['category_name']
    name_raw = info['name_raw']

    df = pd.read_csv(f'{classes[0]}_{classes[1]}_idxs.csv')
    path_to = f"./{category}/raw/"
    all_df = parse_category_special(df, path_to)
    all_df.to_csv(f'genbank_{classes[0]}_{classes[1]}_data.csv', index=False)