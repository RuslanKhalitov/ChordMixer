# ChordMixer: A Scalable Neural Attention Model For Sequences With Different Lengths [Accepted to ICLR'23]

## ChordMixer Architecture 
ChordMixer Network is a stack of ChordMixer blocks. Each of them applies two simple tensor operations on input sequences.
1. **Rotate** step. A parameter-free module that circularly rotates sequence channels.
2. **Mix** step. Applies an MLP over the sequence positions.
<img src="https://github.com/RuslanKhalitov/ChordMixer/blob/main/figures/chordm_nonum_git.gif" width="690">

## Experiments

### Long Range Arena
We get competitive results on a public benchmark. SOTA on **Pathfinder** and **PathfinderX**.


| Model            | ListOps|Text|Image|Retrieval|Pathfinder|PathfinderX|
| -------------    |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Transformer      | 36.37 | 64.27 | 42.44 | 57.46 | 71.40 | ✗ |
| Longformer       | 35.63 | 62.58 | 42.22 | 56.89 | 69.71 | ✗ |
| Linformer        | 37.70 | 53.94 | 38.56 | 52.27 | 76.34 | ✗ |
| Rerformer        | 37.27 | 56.10 | 38.07 | 53.40 | 68.50 | ✗ |
| Performer        | 18.01 | 65.40 | 42.77 | 53.82 | 77.05 | ✗ | 
| Nyströmformer    | 37.15 | 65.52 | 41.58 | 79.56 | 70.94 | ✗ |
| S4               | 59.60 | 86.82 | 88.65 | *90.90* | 94.20 | 96.35 |
| Mega             | **63.14** | **90.43** | **90.44** | **91.25** | 96.01 | 97.98 |
|                  |
| **ChordMixer**   | *[59.89](https://api.wandb.ai/links/rusx/kb9ydn5g)* | *[88.87](https://api.wandb.ai/links/rusx/9e7oizh6)* | *[89.95](https://api.wandb.ai/links/rusx/rk6dt1bt)* | [90.38](https://api.wandb.ai/links/rusx/b9dbjno1) | **[96.67](https://api.wandb.ai/links/rusx/rgi146h9)** | **[98.63](https://api.wandb.ai/links/rusx/empk7dj8)** |

### Insanely long sequences

ChordMixer shows great performance on extremely long sequences with high length variability. We designed experiments with sequences within different domains, such as arithmetic operations, text, and DNA. We demonstrate lengths up to 1.5M in our experiments. 
<img src="https://github.com/RuslanKhalitov/ChordMixer/blob/main/figures/genbank_git.png" width="690">

## Updates
1. [May 2023] Add `ddp` support
2. [May 2023] Add module to calculate and log performance for lengths percentiles 
3. [July 2023] Test and release other models
3. [July 2023] Release pre-training pipeline


## How to use
You can use the ChordMixer backbone directly from this repository. The module does not need any manual cuda kernels. All ChordMixer operations are built-in PyTorch modules. 

ChordMixer can work in two modes: 
* **equal lengths** (sequences have the same lengths or when padding is applied)
* **variable lengths** (sequences have high lengths variability, no padding is applied)

```python
# Equal lengths mode
net = ChordMixer(
    input_size=100,            # Size of the token dict (or size of real-valued input)
    output_size=10,            # Target dim (10 classes)
    embedding_type='sparse',   # 'linear' for real-valued input
    decoder='linear',          # Global average pooling + linear layer
    max_seq_len=2000,          # Maximum sequence length observed in the whole dataset.
    track_size=16,             # Size of tracks to be rotated.
    hidden_size=128,           # Hidden layer size for MLPs.
    mlp_dropout=0.,            # Dropout probability for MLPs.
    layer_dropout=0.,          # Probability for layer dropout.
    prenorm='LN',              # Pre-normalization. One of 'BN', 'LN', 'GN', or 'None' when not applied. 
    norm='LN',                 # Post-normalization. One of 'BN', 'LN', 'GN', or 'None' when not applied. 
    var_len=False              # All sequences are equal in length.
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
    var_len=True                # Use variable length mode
)

lengths = torch.randint(low=1025, high=2048, size=(4, 1)).squeeze()
x = torch.randint(low=1, high=99, size=(torch.sum(lengths), ))
out = net(x, lengths)
print('input size', x.size())
print('output size', out.size())
```


## How to run experiments
Please follow the steps from [the page](../main/chordmixer_instrictions.md) with more examples and running scripts.


## Acknowledgments
This research is funded by The Research Council of Norway.
We want to thank [IDUN group](https://www.hpc.ntnu.no/idun/) for providing resources to complete the experiments. 

Kudos to the HazyResearch team for publicly sharing their well-structured code. The PL training pipelines and the LRA dataloaders in this repo were heavily inspired by their work. 


## Citation
If you use this codebase, datasets, or paper. Please cite us as

```bibtex
@article{khalitov2022chordmixer,
  title={ChordMixer: A Scalable Neural Attention Model for Sequences with Different Lengths},
  author={Khalitov, Ruslan and Yu, Tong and Cheng, Lei and Yang, Zhirong},
  journal={arXiv preprint arXiv:2206.05852},
  year={2022}
}
```


