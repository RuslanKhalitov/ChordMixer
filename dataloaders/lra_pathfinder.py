
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms, datasets
import pytorch_lightning as pl
from PIL import Image
import os
from pathlib import Path

import numpy as np


def LoadGrayscale(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("L")


class PathFinderDataset(Dataset):
    """Path Finder dataset."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        print('DATA DIR', self.data_dir)
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        
        metadata_list = [
            os.path.join(self.data_dir, "metadata", file)
            for file in os.listdir(os.path.join(self.data_dir, "metadata"))
            if file.endswith(".npy")
        ]
        for metadata_file in metadata_list:
            with open(metadata_file, "r") as f:
                for metadata in f.read().splitlines():
                    metadata = metadata.split()
                    image_path = Path(self.data_dir) / metadata[0] / metadata[1]
                    label = int(metadata[3])
                    samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")
            sample = self.transform(sample)
        return sample, target


class Pathfinder(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        num_workers,
        batch_size,
        
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.output_channels = 2

        transform = [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
        self.transform = transforms.Compose(transform)
        
    def setup(self, stage=None):
        dataset = PathFinderDataset(self.data_dir, transform=self.transform)
        len_dataset = len(dataset)
        # LRA Setup
        val_len = int(0.1 * len_dataset)
        test_len = int(0.1 * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x_shape = x.shape
        x = x.view(x_shape[0], x_shape[1], -1)
        x = torch.permute(x, (0, 2, 1))
        batch = x, y
        return batch
    