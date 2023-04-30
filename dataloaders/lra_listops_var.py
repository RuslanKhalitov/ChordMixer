
from pathlib import Path

import pickle
import logging
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, BatchSampler
import pytorch_lightning as pl

import torchtext
from datasets import load_dataset, DatasetDict

from .utils.collators import collate_batch_pad
from .utils.bucket_sampler import BucketSampler

# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()


class ListOpsDataModuleVar(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        max_length=2048,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.cache_dir = self.get_cache_dir()

    def prepare_data(self):
        self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["sequence", "label", "len"])

        # Create all splits
        self.train_dataset, self.val_dataset, self.test_dataset = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )
        self.collate_fn = collate_batch_pad
        
    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )
        dataset = dataset.rename_column("Target", "label")
        
        # Remove unnecessary tokens
        tokenizer = listops_tokenizer
        tokenize = lambda x: {"tokens": tokenizer(x["Source"])}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False
        )
        
        # Calculate lengths
        lengths_calc = lambda x: {"len": len(x["tokens"])}
        dataset = dataset.map(
            lengths_calc,
            keep_in_memory=True,
            load_from_cache_file=False
        )
        
        # Biuld vocab
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(
                ["<pad>", "<unk>"]
            ),
        )
        vocab.set_default_index(vocab["<unk>"])
        
        # Map vocab values
        numericalize = lambda x: {"sequence": vocab(x["tokens"])}
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False
        )

        self._save_to_cache(dataset, tokenizer, vocab)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab):
        cache_dir = self.data_dir / self._cache_dir_name
        os.makedirs(str(cache_dir), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self):
        assert self.cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(self.cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(self.cache_dir))
        with open(self.cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(self.cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"listops1_var_{self.batch_size}"

    def get_cache_dir(self):
        cache_dir = self.data_dir / self._cache_dir_name
        if cache_dir.is_dir():
            return cache_dir
        else:
            return None

    def get_bucket_boundaries(self, df):
        max_log2_bin = math.ceil(np.log2(max(df['len'])))
        min_log2_bin = math.floor(np.log2(min(df['len'])))
        return [2**i for i in range(min_log2_bin, max_log2_bin + 1)]
    
       # Defining DataLoaders
    def train_dataloader(self):
        boundaries = self.get_bucket_boundaries(self.train_dataset)
        sampler_train = BucketSampler(
            lengths=self.train_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler_train,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        return train_dataloader

    def val_dataloader(self):
        boundaries = self.get_bucket_boundaries(self.val_dataset)
        sampler_val = BucketSampler(
            lengths=self.val_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler_val,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        return val_dataloader
    
    def test_dataloader(self):
        sampler_test = BucketSampler(
            lengths=self.test_dataset['len'], 
            bucket_boundaries=self.get_bucket_boundaries(self.test_dataset),
            batch_size=self.batch_size
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler_test,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        return test_dataloader