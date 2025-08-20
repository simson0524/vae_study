# src/dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_loaders(root, name="MNIST", batch_size=128, num_workers=0):
    tfm = transforms.ToTensor()
    dataset = datasets.MNIST if name=="MNIST" else datasets.FashionMNIST
    train = dataset(root, train=True, download=True, transform=tfm)
    test  = dataset(root, train=False, download=True, transform=tfm)

    train_loader= DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader