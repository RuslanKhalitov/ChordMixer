import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from pathlib import Path
from datasets import load_dataset, DatasetDict
import torchtext
import pickle
import logging
from pathlib import Path


class IMDBDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        num_workers,
        batch_size
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = 4096
        self.num_workers = num_workers
        self.vocab_min_freq = 10
        self.append_bos = False
        self.append_eos = True
        self.cache_dir = self.get_cache_dir()
        
    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()

        dataset = load_dataset("imdb", cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        tokenizer = list
        
        max_length = self.max_length - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:max_length]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.vocab_min_freq,
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
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
        return "IMDB"
    
    def get_cache_dir(self):
        cache_dir = self.data_dir / self._cache_dir_name
        if cache_dir.is_dir():
            return cache_dir
        else:
            return None
        
    def prepare_data(self):
        if self.cache_dir is None:
            load_dataset("imdb", cache_dir=self.data_dir)
            self.process_dataset()
            
    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        self.train_dataset, self.test_dataset = dataset["train"], dataset["test"]
        
        # Use test set as val set, as done in the LRA paper
        self.val_dataset = self.test_dataset

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, lengths
        
        self.collate_fn = collate_batch
        print('ID OF PADDING', self.vocab["<pad>"])

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_dataloader
    