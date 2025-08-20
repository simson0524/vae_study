# dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_loader(root, name="MNIST", batch_size=128, num_workers=0, random_seed_1=42, random_seed_2=777):
    tfm = transforms.ToTensor()
    dataset = datasets.MNIST if name=="MNIST" else datasets.FashionMNIST
    train = dataset(root, train=True, download=True, transform=tfm)
    test  = dataset(root, train=False, download=True, transform=tfm)
    
    train_loader_seed_1 = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(random_seed_1)
    )
    train_loader_seed_2 = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(random_seed_2)
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader_seed_1, train_loader_seed_2, test_loader