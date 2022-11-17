from training_utils import count_params, seed_everything, init_weights, train_epoch, eval_model
from chordmixer import ChordMixerNet
from samsa import SAMSA
from dataloader_utils import LRADataset

from sklearn.metrics import accuracy_score
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import math
import numpy as np
import wandb

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem", type=str, default='text_4000')
#parser.add_argument("--problem", type=str, default='retrieval_4000')

#parser.add_argument("--model", type=str, default='chord')
parser.add_argument("--model", type=str, default='samsa')

parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--bs', type=int, default=1)

parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--wandb", type=str, default='leic-no')

args = parser.parse_args()

seed_everything(1)
config = {
    'positional_embedding': False,

    'max_seq_len': 4000,
    'vocab_size': 256,
    'n_class': 2,
    'embedding_type': 'sparse',
    'mlp_dropout': 0.,
    'layer_dropout': 0.,

    'track_size': 16,
    'embedding_size': 240,
    'hidden_size': 240,
    'n_epochs': 200,
}

# config = {
#     'hidden_size': 128,
#     'n_epochs': 270,
#     'positional_embedding': False,
# }

torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)


TASK = args.problem
MODEL = args.model
LR = args.lr
BATCH = args.bs

wandb.init(project="lra", entity=args.wandb, name=TASK + '_' + MODEL + '_bs' + str(BATCH) + '_lr' + str(LR))
wandb.config = config


if MODEL == 'chord':
    net = ChordMixerNet(
        TASK,
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        embedding_size=config['embedding_size'],
        track_size = config['track_size'],
        hidden_size=config['hidden_size'],
        mlp_dropout=config['mlp_dropout'],
        layer_dropout=config['layer_dropout'],
        n_class=config['n_class'],
        positional_embedding=config['positional_embedding'],
        embedding_type=config['embedding_type'],
    )
else:
    net = SAMSA(
        TASK,
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        embedding_size=config['embedding_size'],
        hidden_size=config['hidden_size'],
        mlp_dropout=config['mlp_dropout'],
        layer_dropout=config['layer_dropout'],
        n_class=config['n_class'],
        positional_embedding=config['positional_embedding'],
        embedding_type=config['embedding_type'],
    )


class NetDual(nn.Module):
    def __init__(self, model_part, dim, n_class):
        super(NetDual, self).__init__()
        self.model = model_part
        self.linear = nn.Linear(dim*4, n_class)

    def forward(self, x1, x2):
        y_dim1 = self.model(x1)
        y_dim2 = self.model(x2)
        y_class = torch.cat([y_dim1, y_dim2, y_dim1 * y_dim2, y_dim1 - y_dim2], dim=1)
        y = self.linear(y_class)
        return y


n_tracks = math.ceil(np.log2(config['max_seq_len']))
if MODEL == 'chord':
    DIM = n_tracks * config['track_size']
else:
    DIM = config['embedding_size']

if TASK == 'retrieval_4000':
    net = NetDual(net, DIM, config['n_class'])
else:
    net = net

net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))
print(config)

loss = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    net.parameters(),
    lr=LR,
    betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01
)


trainloader = DataLoader(LRADataset(f'./lra_pickle/{TASK}.train.pickle', True), batch_size=BATCH, shuffle=True, drop_last=False)
testloader = DataLoader(LRADataset(f'./lra_pickle/{TASK}.test.pickle', False), batch_size=BATCH, shuffle=False, drop_last=False)


for epoch in range(config['n_epochs']):
    print(f'Starting epoch {epoch+1}')
    train_epoch(TASK, net, optimizer, loss, trainloader, device=device, log_every=10000)
    with torch.no_grad():
        acc = eval_model(TASK, net, testloader, metric=accuracy_score, device=device)
    print(f'Epoch {epoch+1} completed. Test accuracy: {acc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{acc:.3f}.pt')
