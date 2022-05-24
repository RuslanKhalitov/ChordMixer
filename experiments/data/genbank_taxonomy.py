import pandas as pd
# import pickle
import pickle5 as pickle
from Bio import GenBank
import gzip
import sys
import glob

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


def parse_category(category, genbank_path, dump_seqs=False):
    files = glob.glob("%s/%s/raw/*.seq.gz" % (genbank_path, category))
    n_file = len(files)
    if dump_seqs:
        seqs = []
    taxs = []
    lengths = []
    for fi in range(n_file):
        fname = "%s/%s/raw/%s%d.seq.gz" % (genbank_path, category, prefix[category], fi + 1)
        try:
            with gzip.open(fname, "rt") as handle:
                try:
                    for record in GenBank.parse(handle):
                        if dump_seqs:
                            seqs.append(record.sequence)
                        taxs.append(record.taxonomy)
                        lengths.append(len(record.sequence))
                except:
                    print("Error parsing a record in %s. Skip it." % fname)
            print("fi=%d/%d, n_seq=%d" % (fi+1, n_file, len(lengths)))
        except:
            print("Error parsing %s. Skip it." % fname)
    if dump_seqs:
        pickle.dump(seqs, open("%s/%s/%s_seqs.pkl" % (genbank_path, category, category), "wb"))
    pickle.dump(taxs, open("%s/%s/%s_taxonomies.pkl" % (genbank_path, category, category), "wb"))
    pickle.dump(lengths, open("%s/%s/%s_lengths.pkl" % (genbank_path, category, category), "wb"))


def parse_category_special(category, genbank_path, dump_seqs=False):
    files = glob.glob("%s/%s/raw/*.seq.gz" % (genbank_path, category))
    n_file = len(files)
    if dump_seqs:
        seqs = []
    taxs = []
    lengths = []
    for fi in range(n_file):
        fname = "%s/%s/raw/%s%d.seq.gz" % (genbank_path, category, prefix[category], fi + 1)
        try:
            with gzip.open(fname, "rt") as handle:
                try:
                    for record in GenBank.parse(handle):
                        if dump_seqs:
                            seqs.append(record.sequence)
                        taxs.append(record.taxonomy)
                        lengths.append(len(record.sequence))
                except:
                    print("Error parsing a record in %s. Skip it." % fname)
            print("fi=%d/%d, n_seq=%d" % (fi+1, n_file, len(lengths)))
        except:
            print("Error parsing %s. Skip it." % fname)
    if dump_seqs:
        pickle.dump(seqs, open("%s/%s/%s_seqs.pkl" % (genbank_path, category, category), "wb"))
    pickle.dump(taxs, open("%s/%s/%s_taxonomies.pkl" % (genbank_path, category, category), "wb"))
    pickle.dump(lengths, open("%s/%s/%s_lengths.pkl" % (genbank_path, category, category), "wb"))

class PhylogeneticTreeNode:
    def __init__(self, tax_id=None, rank=None, parent=None):
        self.tax_id = tax_id
        self.rank = rank
        self.name_nodes = []
        self.parent = parent
        self.children = []


class NameNode:
    def __init__(self, tax_name=None, tree_node=None):
        self.tax_name = tax_name
        self.tree_node = tree_node
        self.count = 0


class GenBankTaxonomy:
    def __init__(self, genbank_path):
        self.genbank_path = genbank_path
        self.df_nodes = pd.read_pickle(genbank_path + "/taxdump/df_nodes.pkl")
        self.df_names = pd.read_pickle(genbank_path + "/taxdump/df_names.pkl")
        self.ranks = self.df_nodes['rank'].unique()
        self.rank_classes = self.reset_rank_classes()
        self.df_nodes['tax_id'] = self.df_nodes['tax_id'].astype(int)
        self.df_nodes['parent_tax_id'] = self.df_nodes['parent_tax_id'].astype(int)
        self.df_names['tax_id'] = self.df_names['tax_id'].astype(int)
        self.df_names['name_txt'] = self.df_names['name_txt'].str.casefold()
        self.dict_tax_id = {}
        self.dict_tax_name = {}
        self.tree = self.build_phylogenetic_tree()

    def name2taxids(self, tax_name):
        rows = self.df_names.loc[self.df_names['name_txt'].isin([tax_name.casefold()])]
        return rows['tax_id'].values

    def name2ranks(self, tax_name):
        taxids = self.name2taxids(tax_name)
        rows = self.df_nodes.loc[self.df_nodes['tax_id'].isin(taxids)]
        return rows['rank'].values

    def append_name_to_rank(self, name, rank):
        self.rank_classes[rank].add(name)

    def build_phylogenetic_tree(self):
        print("Preparing tree nodes")
        for i, row in self.df_nodes.iterrows():
            if i % 100000 == 0:
                print("i=%d/%d" % (i, len(self.df_nodes)))
            tax_id = row['tax_id']
            tree_node = PhylogeneticTreeNode(tax_id=tax_id, rank=row['rank'])
            self.dict_tax_id[tax_id] = tree_node

        root = None
        print("Constructing tree")
        for i, row in self.df_nodes.iterrows():
            if i % 100000 == 0:
                print("i=%d/%d" % (i, len(self.df_nodes)))
            tax_id = row['tax_id']
            parent_tax_id = row['parent_tax_id']
            try:
                this_node = self.dict_tax_id[tax_id]
                parent_node = self.dict_tax_id[parent_tax_id]
                this_node.parent = parent_node
                parent_node.children.append(this_node)

                if tax_id == parent_tax_id:
                    root = this_node
            except KeyError:
                print("parent_tax_id %d not found")

        print("Linking names")
        for i, row in self.df_names.iterrows():
            if i % 100000 == 0:
                print("i=%d/%d" % (i, len(self.df_names)))
            tax_name = row['name_txt']
            name_node = NameNode(tax_name)
            self.dict_tax_name[tax_name] = name_node
            tax_id = row['tax_id']
            tree_node = self.dict_tax_id[tax_id]
            name_node.tree_node = tree_node
            tree_node.name_nodes.append(name_node)

        return root

    def reset_name_counts(self):
        for nn in self.dict_tax_name.values():
            nn.count = 0

    def reset_rank_classes(self):
        return {rank: set() for rank in self.ranks}

    def taxonomies2classes(self, category):
        all_tax_names = pickle.load(open("%s/%s/%s_taxonomies.pkl" % (self.genbank_path, category, category), "rb"))
        dict_name2ranks = {}
        n_seq = len(all_tax_names)
        self.rank_classes = self.reset_rank_classes()
        print("===== Mapping taxonomies to ranks =========")
        for i_seq, seq_tax_names in enumerate(all_tax_names):
            if i_seq % 100000 == 0:
                print("%d/%d" % (i_seq, n_seq))
            for tax_name in seq_tax_names:
                if tax_name in dict_name2ranks:
                    for rank in dict_name2ranks[tax_name]:
                        self.append_name_to_rank(tax_name, rank)
                else:
                    ranks = self.name2ranks(tax_name)
                    dict_name2ranks[tax_name] = ranks
                    for rank in ranks:
                        self.append_name_to_rank(tax_name, rank)

        records = []
        print("===== Creating classes dataframe ==========")
        for i_seq, seq_tax_names in enumerate(all_tax_names):
            if i_seq % 100000 == 0:
                print("%d/%d" % (i_seq, n_seq))

            seq_classes = {rank: 'unknown' for rank in self.ranks}
            for tax_name in seq_tax_names:
                ranks = dict_name2ranks[tax_name]
                for rank in ranks:
                    seq_classes[rank] = tax_name
            records.append(seq_classes)

        df_classes = pd.DataFrame.from_records(records)
        return df_classes

    def count_names(self, category):
        all_tax_names = pickle.load(open("%s/%s/%s_taxonomies.pkl" % (self.genbank_path, category, category), "rb"))
        n_seq = len(all_tax_names)
        not_found_names = set()
        for i_seq, seq_tax_names in enumerate(all_tax_names):
            if i_seq % 100000 == 0:
                print("%d/%d" % (i_seq, n_seq))
            for tax_name in seq_tax_names:
                try:
                    nn = self.dict_tax_name[tax_name.casefold()]
                    nn.count += 1
                except KeyError:
                    not_found_names.add(tax_name)
        print("The following names are not found in the GenBank Tree:")
        print(not_found_names)


def main():
    genbank_path = "./"
    gbt = GenBankTaxonomy(genbank_path)
    pickle.dump(gbt, open("%s/gbt.pkl" % genbank_path, "wb"))


if __name__ == '__main__':
    main()
