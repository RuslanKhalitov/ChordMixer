import os

prefix = {
    # 'Bacterial': 'gbbct',
    # 'Invertebrate': 'gbinv',
    'Other mammalian': 'gbmam',
    # 'Plant': 'gbpln',
    'Primate': 'gbpri',
    'Rodent': 'gbrod',
    # 'Viral': 'gbvrl',
    'Other vertebrate': 'gbvrt'
}

# Date: accessed on 01.11.2022
# Time needed to get the raw data: ~4h

n_seq_per_category = {
    'Bacterial': 688,
    'Invertebrate': 488,
    'Other mammalian': 116,
    'Plant': 728,
    'Primate': 56,
    'Rodent': 77,
    'Viral': 375,
    'Other vertebrate': 277
}

import genbank_taxonomy

# STEP 1.
# Saves raw sequences to the corresponding folders
def download_raw():
    for category in prefix.keys():
        n_seq = n_seq_per_category[category]
        for i in range(n_seq):
            fi = i + 1
            cmd = "wget -P '%s/raw' https://ftp.ncbi.nih.gov/genbank/%s%d.seq.gz" % (category, prefix[category], fi)
            print(cmd)
            os.system(cmd)


# STEP 2.
# Parses raw data. Saves statistics
def parse_raw(genbank_path):
    for category in prefix.keys():
        genbank_taxonomy.parse_category(category, genbank_path, dump_seqs=True)
    
    
if __name__ == "__main__":
    download_raw()
    parse_raw(genbank_path='.')