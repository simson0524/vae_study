# src/ae_train.py

from src.loss import reconst_loss
from tqdm.auto import tqdm
import torch
import os

def ae_train(model, train_loader, test_loader, config, device="cuda"):
    model.train()

    device = torch.device(config['device'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    out_dir = config['paths']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, config['train']['epochs']+1):
        total = 0
        for x, _ in tqdm(train_loader, total=len(train_loader), desc=f"[EPOCH {epoch}] Training..."):
            x = x.to(device)
            # forward
            reconst_x, _, _, latent_z = model(x)

            # loss
            loss = reconst_loss(reconst_x, x)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"[EPOCH {epoch}] loss={total/len(train_loader):.4f}")