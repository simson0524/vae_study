# src/visualization.py

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch
import os

def save_reconst(model, loader, out_dir, n=32, tag="model", device="cuda"):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    x, _ = next(iter(loader))
    x = x.to(device)[:n]
    with torch.inference_mode():
        out = model(x)
        xhat = out[0] if isinstance(out, tuple) else out
    grid = torch.cat([x, xhat], 0).clamp(0, 1)
    save_image(grid.cpu(), f"{out_dir}/{tag}_recon.png", nrow=n)



def save_latent_scatter(model, loader, out_dir, latent_dim, tag="vae"):
    pass

# 어떤 latent dim으로 하면 잘 되는지 실험 한번 가봄