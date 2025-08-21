# main.py

from src.utils import load_config, set_seed
from src.dataset import get_loaders
from src.model import ConvVAE, ConvAE
from src.vae_train import vae_train, ae_train
import argparse
import sys
from pathlib import Path

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args, _ = parser.parse_known_args(argv)
    return args

def main():
    args = parse_args(sys.argv[1:])
    cfg_path = Path(args.config).expanduser().resolve()
    config = load_config(str(cfg_path))

    set_seed(config['seed'])

    train_loader, test_loader = get_loaders(
        root=config['data']['root'],
        name=config['data']['name'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
    )

    vae_model = ConvVAE(latent_dim=config['model']['latent_dim'])
    vae_train(vae_model, train_loader, test_loader, config)

    ae_model = ConvAE(latent_dim=config['model']['latent_dim'])
    ae_train(ae_model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()