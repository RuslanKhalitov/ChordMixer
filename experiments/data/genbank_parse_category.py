import genbank_taxonomy


genbank_path = '.'
for category in ['Bacterial', 'Invertebrate', 'Other mammalian', 'Plant', 'Primate', 'Rodent', 'Viral', 'Other vertebrate']:
    genbank_taxonomy.parse_category(category, genbank_path, dump_seqs=True)
