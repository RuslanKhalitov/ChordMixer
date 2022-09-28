# This is the experiment setup we used to report models' performance in our paper.

Experiment consists of three tasks:

- (*adding*) Adding Problem with variable length
- (*genbank*) DNA-based taxonomy classification
- (*longdoc*) Long Documents Classifications

The adding and genbank tasks consist of sublevels. The base sequence length for adding (200, 1000, 16000, and 128000) and specific genuses for genbank (Sus vs. Bos, Danio vs. Cyprinus, Mus vs. Rattus, and Carassius vs. Labeo). The longdoc task is unique.

## Adding problem

Adding problem is a synthetic dataset that does not require additional sources.
To generate the data you need to run adding_data_generator.py and set the base sequence length parameter.
As a result it will output training/validation/testing tensors.

## Genbank problem

This dataset cosists of DNA sequences downloaded from genbank.
The preprocessing includes the following steps:
- Download all the genbank sequences
- Group them to the family pkl files
- Sample sequences to construct the corresponding training problems
- Save final .pkl files

## Longdoc problem

Sequences in this dataset are made of academic papers from 4 subjects.
The articles are publicly available.
To get the final files you need to run longdoc_preprocessing.py
- It will download the source data from github
- Preprocess them to construct csv files
- Read and make a multiclass classification problem
- Save filan .pkl files
