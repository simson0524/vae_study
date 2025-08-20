# visualization.py

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch
import os


def save_reconst(model, loader, out_dir, n=32, tag="vae"):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(next(model.paramenters()).device)[:n]
    with torch.no_grad():
        out = model(x)
        reconst_x = out[0] if isinstance(out, tuple) else out
    grid = torch.cat([x, reconst_x], 0)
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid.cpu(), f"{out_dir}/{tag}_reconst.png", nrow=n)


def save_latent_scatter(model, loader, out_dir, latent_dim=4, tag="vae"):
