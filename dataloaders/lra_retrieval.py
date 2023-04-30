from pathlib import Path

import pickle
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

import torchtext
from datasets import load_dataset, DatasetDict, Value

class AANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        num_workers,
        batch_size,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_length = 4000
        self.append_bos = False
        self.append_eos = True
        self.l_max = 4000
        self.n_workers = num_workers
        self.cache_dir = self.get_cache_dir()

    def prepare_data(self):
        self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs1, xs2, ys = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label"])
                    for data in batch
                ]
            )
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L-xs1.size(1)), value=self.vocab["<pad>"])
            xs2 = F.pad(xs2, (0, L-xs2.size(1)), value=self.vocab["<pad>"])
            ys = torch.tensor(ys)
            return xs1, xs2, ys, lengths1, lengths2

        self._collate_fn = collate_batch

    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )
        
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        encode = lambda text: vocab(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
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
        return f"aan_{self.max_length}_{self.append_bos}_{self.append_eos}"

    def get_cache_dir(self):
        cache_dir = self.data_dir / self._cache_dir_name
        if cache_dir.is_dir():
            return cache_dir
        else:
            return None
        
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self._collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
        return test_dataloader