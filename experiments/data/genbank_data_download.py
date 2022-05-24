import os

prefix = {
    'Bacterial': 'gbbct',
    'Invertebrate': 'gbinv',
    'Other mammalian': 'gbmam',
    'Plant': 'gbpln',
    'Primate': 'gbpri',
    'Rodent': 'gbrod',
    'Viral': 'gbvrl',
    'Other vertebrate': 'gbvrt'
}

n_seq = [688, 488, 116, 728, 56, 77, 375, 277]  # Date: 16.02.2022

category = 'Rodent'
for i in range(len(n_seq)):
    fi = i + 1
    cmd = "wget -P '%s/raw' https://ftp.ncbi.nih.gov/genbank/%s%d.seq.gz" % (category, prefix[category], fi)
    print(cmd)
    os.system(cmd)