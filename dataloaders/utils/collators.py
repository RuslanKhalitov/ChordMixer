import torch 
import torch.nn as nn


def collate_batch_pad(batch, pad_value=0):
    xs, ys, lengths = batch["sequence"], batch["label"], batch["len"]
    xs = nn.utils.rnn.pad_sequence(
        xs, padding_value=pad_value, batch_first=True
    )
    return xs, ys, lengths

def collate_batch(batch, pad_value=0):
    xs, ys = zip(*[(data["sequence"], data["label"]) for data in batch])
    lengths = torch.tensor([len(x) for x in xs])
    xs = nn.utils.rnn.pad_sequence(
        xs, padding_value=pad_value, batch_first=True
    )
    ys = torch.tensor(ys)
    return xs, ys, lengths

# def collate_batch_horizontal(batch):
#     xs, ys, lengths = zip(*[(data["sequence"], data["label"], data["len"]) for data in batch])
#     xs = torch.cat(xs, 0)
#     ys = torch.tensor(ys)
#     return xs, ys, list(lengths)

def collate_batch_horizontal(batch):
    xs, ys, lengths = batch["sequence"], batch["label"], batch["len"]
    if isinstance(xs, torch.Tensor):
        # happens when we diplicate a single sequence
        xs = torch.flatten(xs)
    else:
        xs = torch.cat(xs, 0)
    return xs, ys, lengths

def collate_batch_horizontal_adding(batch):
    xs, ys, lengths = batch["sequence"].values, batch["label"].values, batch["len"].values
    xs = [torch.tensor(x, dtype=torch.float32) for x in xs]
    ys = torch.tensor(ys, dtype=torch.float32)
    lengths = torch.tensor(lengths)
    if isinstance(xs, torch.Tensor):
        xs = torch.flatten(xs)
    else:
        xs = torch.cat(xs, 0)
    return xs, ys, lengths