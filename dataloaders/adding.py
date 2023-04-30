
from pathlib import Path
import math
import torch
import pickle
import pandas as pd
import logging
import os
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torchtext
from datasets import load_dataset, DatasetDict, Dataset

from .utils.bucket_sampler import BucketSampler
from .utils.collators import collate_batch_horizontal_adding

class DatasetAdding(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
            
    def __getitem__(self, index):
        items = self.df.iloc[index, :]
        return items

    def __len__(self):
        return len(self.df)
    
class AddingDataModule(pl.LightningDataModule):
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

    # def prepare_data(self):
    #     self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        
        self.train_dataset = pd.read_pickle(str(self.data_dir / f"{self.taskname}_train.pkl")).reset_index(drop=True)
        self.test_dataset = pd.read_pickle(str(self.data_dir / f"{self.taskname}_test.pkl")).reset_index(drop=True)
        self.val_dataset = pd.read_pickle(str(self.data_dir / f"{self.taskname}_val.pkl")).reset_index(drop=True)

        self.collate_fn = collate_batch_horizontal_adding
    

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
        train_adding = DatasetAdding(df=self.train_dataset)
        sampler_train = BucketSampler(
            lengths=self.train_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        train_dataloader = DataLoader(
            train_adding,
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
        val_adding = DatasetAdding(df=self.test_dataset)
        sampler_val = BucketSampler(
            lengths=self.test_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        val_dataloader = DataLoader(
            val_adding,
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
        test_adding = DatasetAdding(df=self.test_dataset)
        test_dataloader = DataLoader(
            test_adding,
            batch_size=None,
            shuffle=False,
            sampler=sampler_test,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        return test_dataloader