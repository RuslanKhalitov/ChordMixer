import torch
from torch import nn
import logging
import os
import random
import numpy as np
from tqdm import tqdm

def init_weights(m):
    """
    Xavier weights initialization for linear layers
    :param m: model parameters
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_logger(args, training_config, model_config):
    if args.logger == 'file':
        filename = f'{args.problem_class}_{args.problem}_{args.model}.txt'
        os.makedirs("logfiles", exist_ok=True)
        log_file_name = "./logfiles/" + filename
        handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
        loginf = logging.info
        loginf(log_file_name)
        return loginf

    elif args.logger == 'wandb':
        pass