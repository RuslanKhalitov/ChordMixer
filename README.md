# ChordMixer
The official implementation of the ChordMixer architecture.

Arxiv preprint: https://arxiv.org/abs/2206.05852


<img width="800" alt="Screenshot 2022-06-15 at 17 17 12" src="https://user-images.githubusercontent.com/22999405/173863802-c4477a1b-96ec-4e37-83b6-2b128f7d6c26.png">


## Standalone ChordMixer implementation

To apply ChordMixer architecture on your datasets, please use both chordmixer.py and dataloader_utils.py to ensure the correct batch construction.

## Datasets

Follow the instructions from /experiments/README.md to get the datasets from this paper.
All the datasets should be stored in the experiments/data folder.

## Demo 
We added the demonstration of the ChordMixer training process in chordmixer_demo.ipynb
You can run training of ChordMixer with different datasets. You need to setup wandb account to see the detailed logging and performance results within length percentiles.


```
cd experiments
python3 train_adding.py --problem_class adding --problem 200 --model chordmixer --device_id 0 --wandb %yourusername%
python3 train_adding.py --problem_class adding --problem 1000 --model chordmixer --device_id 0 --wandb %yourusername%
python3 train_adding.py --problem_class adding --problem 16000 --model chordmixer --device_id 0 --wandb %yourusername%
python3 train_adding.py --problem_class adding --problem 12800 --model chordmixer --device_id 0 --wandb %yourusername%

python3 train_genbank.py --problem_class genbank --problem 'Carassius vs. Labeo' --model chordmixer --device_id 0 --wandb %yourusername%
python3 train_genbank.py --problem_class genbank --problem 'Sus vs. Bos' --model chordmixer --device_id 0 --wandb %yourusername%
python3 train_genbank.py --problem_class genbank --problem 'Mus vs. Rattus' --model chordmixer --device_id 0 --wandb %yourusername%
python3 train_genbank.py --problem_class genbank --problem 'Danio vs. Cyprinus' --model chordmixer --device_id 0 --wandb %yourusername%

python3 train_longdoc.py --problem_class longdoc --problem longdoc --model chordmixer --device_id 0 --wandb %yourusername%
```



