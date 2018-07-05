import random
import torch
import numpy as np
import yaml


def read_config(filepath):
    with open(filepath, 'r') as stream:
        config = yaml.load(stream)
    return config


def set_global_seeds(i):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)
