# This is the experiment setup we used to report models' performance in our paper.

Experiment consists of three tasks:

- (*adding*) Adding Problem with variable length
- (*genbank*) DNA-based taxonomy classification
- (*longdoc*) Long Documents Classifications

The adding and genbank tasks consist of sublevels. The base sequence length for adding (200, 1000, 16000, and 128000) and specific genuses for genbank (Sus vs. Bos, Danio vs. Cyprinus, Mus vs. Rattus, and Carassius vs. Labeo). The longdoc task is unique.

## Adding problem

Adding problem is a synthetic dataset that does not require additional sources.
To generate the data you need to set the base sequence length parameter {200, 1000, 16000, 128000} and run adding_data_generator.py.
As a result it will output training/validation/testing files.

Expected output:
- data/adding_{base_length}_train.pkl
- data/adding_{base_length}_val.pkl 
- data/adding_{base_length}_test.pkl
Max_size: ~50Gb for 128k, ~32Gb for 16k

## Genbank problem

This dataset cosists of DNA sequences downloaded from genbank.
The preprocessing includes the following steps:
- Download all the genbank sequences ('genbank_data_download.py'). Saves them as *.seq.gz.
- Parse the category ('genbank_parse_category.py'). Save taxonomy files as *_taxonomies.pkl 
- Map the taxonomy category to a class ('genbank_tax2class.py')
- Filter out the unnecessary caterories and saves sequences and classes into *.csv files ('genbank_create_csv.py')
- For each task split data into train/test ('genbank_preprocessing.py'). Save train/test files as *.pkl 

Expected output: 
- data/danio_cyprinus_train.pkl
- data/danio_cyprinus_test.pkl
- data/mus_rattus_train.pkl
- data/mus_rattus_test.pkl
- data/carassius_labeo_train.pkl
- data/carassius_labeo_test.pkl
- data/sus_bos_train.pkl
- data/sus_bos_test.pkl
Max_size: ~2Gb

## Longdoc problem

Sequences in this dataset are made of academic papers from 4 subjects.
The articles are publicly available.
1. You have to download .rar archives manually from https://github.com/LiqunW/Long-document-dataset and put them to the \data folder, or 
use download_data() function from longdoc_processing.py
2. Extract data from .rar files. You will see the folders with articles from the corresponding sections.

Expected folders structure: 
- data/cs.AI/*.txt
- data/cs.NE/*.txt
- data/math.GR/*.txt
- data/math.AC/*.txt

To get the final files you need to run longdoc_preprocessing.py
- It preprocess source .txt files to construct .csv-s
- Concats them to a single dataframe
- Splits the data into train/(val)/test 
- Saves final .pkl files

Expected output: 
- data/longdoc_train.pkl
- data/longdoc_test.pkl
Max_size: ~200Mb each