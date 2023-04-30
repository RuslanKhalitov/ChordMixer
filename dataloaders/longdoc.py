
from pathlib import Path
import math
import pickle
import pandas as pd
import logging
import os
import numpy as np
from random import shuffle
from torch.utils.data import Sampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

import torchtext
from datasets import load_dataset, DatasetDict

from .utils.bucket_sampler import BucketSampler
from .utils.preprocess_longdoc import PrepareLongdoc
from .utils.collators import collate_batch_horizontal, collate_batch_pad

class LongdocDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        taskname,
        num_workers,
        batch_size,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.taskname = taskname
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = self.get_cache_dir()

    def prepare_data(self):
        self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset = self.process_dataset()
        dataset.set_format(type="torch", columns=["sequence", "label", "len"])

        # Create all splits
        self.train_dataset, self.test_dataset = (
            dataset["train"],
            dataset["test"],
        )
        self.collate_fn = collate_batch_horizontal
    
    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()
        
        # Create train and test csv
        file = self.data_dir / "longdoc_train.csv"
        if not file.exists():
            PrepareLongdoc(self.data_dir)
        
        # Read train and test csv
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "longdoc_train.csv"),
                "test": str(self.data_dir / "longdoc_test.csv")
            },
            keep_in_memory=True,
        )
        
        tokenizer = list
        tokenize = lambda example: {"tokens": tokenizer(example["content"])}
        dataset = dataset.map(
            tokenize,
            remove_columns=["content"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(["<pad>", "<unk>"]),
        )
        vocab.set_default_index(vocab["<pad>"])
        
        numericalize = lambda example: {
            "sequence": vocab(example["tokens"])
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        
        self._save_to_cache(dataset)
        return dataset

    def _save_to_cache(self, dataset):
        cache_dir = self.data_dir / self._cache_dir_name
        os.makedirs(str(cache_dir), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))

    def _load_from_cache(self):
        assert self.cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(self.cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(self.cache_dir))
        return dataset

    @property
    def _cache_dir_name(self):
        return f"{self.taskname}"

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
            sampler=sampler_train,# sampler or batch sampler
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        return train_dataloader

    def val_dataloader(self):
        boundaries = self.get_bucket_boundaries(self.test_dataset)
        sampler_val = BucketSampler(
            lengths=self.test_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        val_dataloader = DataLoader(
            self.test_dataset,
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