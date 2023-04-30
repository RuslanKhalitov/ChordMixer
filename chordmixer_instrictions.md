# Here we describe in detail how to use ChordMixer codebase

## How to use
You can use the ChordMixer backbone directly from this repository. The module does not need any manual cuda kernels. All ChordMixer operations are built-in PyTorch modules. 

ChordMixer can work in two modes: 
* **equal lengths** (sequences have the same lengths or when padding is applied)
* **variable lengths** (sequences have high lengths variability, no padding is applied)

### Encoder
#### Equal lengths mode
```python
import torch
import numpy as np
import math
from models.chordmixer import ChordMixerEncoder, ChordMixer

encoder = ChordMixerEncoder(
    max_seq_len=2048,  # Maximum sequence length observed in the whole dataset.
    track_size=16,     # Size of tracks to be rotated.
    hidden_size=128,   # Hidden layer size for MLPs.
    mlp_dropout=0.,    # Dropout probability for MLPs.
    layer_dropout=0.,  # Probability for layer dropout.
    prenorm='LN',      # Pre-normalization. One of 'BN', 'LN', 'GN', or 'None' when not applied. 
    norm='LN',         # Post-normalization. One of 'BN', 'LN', 'GN', or 'None' when not applied. 
    var_len=False      # Use variable length mode.
)

bs = 4
track_size = 16
max_seq_len = 2000
dim = math.ceil(np.log2(max_seq_len) + 1) * track_size
seq_len = 2000
x = torch.randn(size=(bs, max_seq_len, dim))


out = encoder(x)
print('Input size', x.size())
print('Output size', out.size())
```

#### Variable lengths mode
```python


encoder = ChordMixerEncoder(
    max_seq_len=2000,
    track_size=16,
    hidden_size=128,
    mlp_dropout=0.,
    layer_dropout=0.,
    prenorm='None',      # No normalization for variable lengths experiments
    norm='None',
    var_len=True
)

# We concatenate sequences in a batch vertically forming a long sequence.
bs = 4
track_size = 16
max_seq_len = 2000
dim = math.ceil(np.log2(max_seq_len) + 1) * track_size

lengths = torch.randint(low=int(max_seq_len/2), high=max_seq_len, size=(bs, ))
x = torch.randn(size=(sum(lengths), dim))
out = encoder(x, lengths)

print('lengths', lengths)
print('sum of lengths', torch.sum(lengths))
print('input size', x.size())
print('output size', out.size())

```

### End-to-end
```python
# Equal lengths mode
net = ChordMixer(
    input_size=100,            # size of the token dict (or size of real-valued input)
    output_size=10,            # target dim (10 classes)
    embedding_type='sparse',   # 'linear' for real-valued input
    decoder='linear',          # global average pooling + linear layer
    max_seq_len=2000,
    track_size=16,
    hidden_size=128,
    mlp_dropout=0.,
    layer_dropout=0.,
    prenorm='LN',
    norm='LN',
    var_len=False
)

x = torch.randint(low=1, high=99, size=(4, 2000))
out = net(x)
print('input size', x.size())
print('output size', out.size())

# Variable lengths mode
net = ChordMixer(
    input_size=100,
    output_size=10,
    embedding_type='sparse',
    decoder='linear',
    max_seq_len=2000,
    track_size=16,
    hidden_size=128,
    mlp_dropout=0.,
    layer_dropout=0.,
    prenorm='None',
    norm='None',
    var_len=True
)

lengths = torch.randint(low=1025, high=2048, size=(4, 1)).squeeze()
x = torch.randint(low=1, high=99, size=(torch.sum(lengths), ))
out = net(x, lengths)
print('input size', x.size())
print('output size', out.size())
```

### Bucket Sampler
For correct work in the varibale sequences mode you need to ensure sequences are within the same log2 group. See details in the paper.

For example, `(32, 64]` or `(65536, 131072]`.

We made a custom sampler to ensure the right sampling.

```python
from dataloaders.utils.bucket_sampler import BucketSampler

sequence_lengths = [random.randint(32, 65536) for i in range(2000)] 
min_log2_bin = math.ceil(np.log2(min(sequence_lengths)))
max_log2_bin = math.ceil(np.log2(max(sequence_lengths)))
bucket_boundaries = [2**i for i in range(min_log2_bin, max_log2_bin + 1)]

sampler = BucketSampler(
    lengths=sequence_lengths,
    bucket_boundaries=bucket_boundaries,
    batch_size=3
)

it = iter(sampler)
ids = next(it)
print('indices', ids)
print('corresponding lengths', [sequence_lengths[i] for i in ids])
```

## How to run experiments from the paper

0. Make sure all the packages from requirements.txt are installed
1. log into wandb `export WANDB_API_KEY={your_api_key}`. All results are logged to wandb
1. Download or generate datasets. See the instruction in data/README.md
2. Once the data is on the disk, you're ready to run the experiments

> Warning. We used DataParallel mode from PytorchLightning for LRA experiments and a single-gpu mode for the experiments with variable lengths (config sets all devices by default). Please make sure you have the right version of pytorch lightning (requirements.txt). 

```python

DATA=/path/to/data-dir #e.g. ChordMixer/data/genbank
python trainer.py +problem=Carassius_Labeo +model=chordmixer +metric=rocauc +data_dir=${DATA} +diff_len=True +n_devices=1
python trainer.py +problem=Sus_Bos +model=chordmixer +metric=rocauc +data_dir=${DATA} +diff_len=True +n_devices=1
python trainer.py +problem=Mus_Rattus +model=chordmixer +metric=rocauc +data_dir=${DATA} +diff_len=True +n_devices=1
python trainer.py +problem=Danio_Cyprinus +model=chordmixer +metric=rocauc +data_dir=${DATA} +diff_len=True +n_devices=1

DATA=/path/to/data-dir #e.g. ChordMixer/data/longdoc
python trainer.py +problem=longdoc +model=chordmixer +metric=acc +loss=crossentropy +data_dir=${DATA} +diff_len=True +n_devices=1

DATA=/path/to/data-dir #e.g. ChordMixer/data/adding
python trainer.py +problem=adding_200 +model=chordmixer +metric=reg_acc +loss=mse +data_dir=${DATA} +diff_len=True +n_devices=1
python trainer.py +problem=adding_1000 +model=chordmixer +metric=reg_acc +loss=mse +data_dir=${DATA} +diff_len=True +n_devices=1
python trainer.py +problem=adding_16000 +model=chordmixer +metric=reg_acc +loss=mse +data_dir=${DATA} +diff_len=True +n_devices=1
python trainer.py +problem=adding_128000 +model=chordmixer +metric=reg_acc +loss=mse +data_dir=${DATA} +diff_len=True +n_devices=1

DATA=/path/to/data-dir #e.g. ChordMixer/data/lra_release
python trainer.py +problem=image +model=chordmixer +metric=acc +data_dir=${DATA} +loss=crossentropy +diff_len=False
python trainer.py +problem=text +model=chordmixer +metric=acc +data_dir=${DATA} +loss=crossentropy +diff_len=False

DATA=/path/to/data-dir #e.g. ChordMixer/data/lra_release/lra_release/listops-1000
python trainer.py +problem=listops +model=chordmixer +metric=acc +data_dir=${DATA} +loss=crossentropy +diff_len=False

DATA=/path/to/data-dir #e.g. ChordMixer/data/lra_release/lra_release/tsv_data
python trainer.py +problem=retrieval +model=chordmixer +metric=acc +data_dir=${DATA} +loss=crossentropy +diff_len=False

DATA=/path/to/data-dir #e.g. ChordMixer/data/lra_release/lra_release/pathfinder32/curv_contour_length_14/
python trainer.py +problem=pathfinder +model=chordmixer +metric=acc +data_dir=${DATA} +loss=crossentropy +diff_len=False

DATA=/path/to/data-dir #e.g. ChordMixer/data/lra_release/lra_release/pathfinder128/curv_contour_length_14/
python trainer.py +problem=pathfinderx +model=chordmixer +metric=acc +data_dir=${DATA} +loss=crossentropy +diff_len=False

```

