# src/vae_train.py

from src.loss import reconst_loss, beta, kl_divergence
from tqdm.auto import tqdm
import torch
import os

def vae_train(model, train_loader, test_loader, config, device="cuda"):
    model.train()

    device = torch.device(config['device'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    out_dir = config['paths']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, config['train']['epochs']+1):
        total = 0
        reconst_total = 0
        for x, _ in tqdm(train_loader, total=len(train_loader), desc=f"[EPOCH {epoch}] Training..."):
            x = x.to(device)
            # forward
            reconst_x, mu, logvar, latent_z = model(x)

            # loss
            reconst_loss_term = reconst_loss(reconst_x, x)
            beta_term = beta(config, epoch)
            kl_divergence_term = kl_divergence(mu, logvar)
            loss = reconst_loss_term + beta_term*kl_divergence_term

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            reconst_total += reconst_loss_term.item()

        print(f"[EPOCH {epoch}] loss={total/len(train_loader):.4f} (beta={beta_term:.2f})")
        print(f"[EPOCH {epoch}] reconst_loss={reconst_total/len(train_loader):.4f} (beta={beta_term:.2f})\n")