# src/utils.py

import random
import torch
import yaml
import os

def load_config(path="config/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)