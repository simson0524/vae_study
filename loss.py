# loss.py

import torch.nn.functional as F
import torch
import math

def reconst_loss(reconst_x, x):
    loss = F.binary_cross_entropy(reconst_x, x, reduction='sum')
    return loss / x.size(0) # sum of loss / batch_size

def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum( 1+logvar-mu.pow(2)-logvar.exp() )
    return kl / mu.size(0) # sum of kl / batch_size

def beta(config, epoch):
    if config['beta_schedule']['type'] == 'linear':
        t = min(1.0, epoch/max(1, config['beta_schedule']['warmup_epochs']))
        return 1.0 + t*(config['beta_schedule']['warmup_epochs'])
    elif config['beta_schedule']['type'] == 'cosine':
        T = max(1.0, config['beta_schedule']['warmup_epochs'])
        t = min(1.0, epoch / T)
        return 1.0 + 0.5(1-math.cos(math.pi * t))*(config['beta_schedule']['max_beta'] - 1.0)
    return config['model']['beta']