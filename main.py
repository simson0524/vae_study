# main.py

from src.utils import load_config, set_seed
from src.dataset import get_loaders
from src.model import ConvVAE, FlattenVAE
from src.vae_train import vae_train
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['seed'])

    train_loader, test_loader = get_loaders(
        root=config['data']['root'],
        name=config['data']['name'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
    )

    model = ConvVAE(latent_dim=config['model']['latent_dim'])
    vae_train(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()