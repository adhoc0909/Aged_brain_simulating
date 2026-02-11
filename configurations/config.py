import argparse
from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class Config:
    """Configuration for Brain Aging Simulation model."""

    # Model hyperparameters
    lr: float = 0.0001
    decay: float = 0.0001
    latent_space: int = 130
    batch_size: int = 32
    input_shape: Tuple[int, int, int] = (208, 160, 1)
    filters: int = 32
    age_dim: int = 100
    beta_1: float = 0.5
    beta_2: float = 0.999

    # Training parameters
    critic_iter: int = 5
    use_cuda: bool = True
    gp_weight: float = 10.0
    id_weight: float = 100.0
    self_rec_weight: float = 10.0
    epochs: int = 600

    # Paths (relative paths as defaults)
    data_root: str = "./data/"
    paired_list_csv: str = "./paired_image_files.csv"
    model_save_path: str = "./result/model/"
    sample_image_save_path: str = "./result/sample_image/"


def parse_args() -> Config:
    """Parse command line arguments and return Config object."""
    parser = argparse.ArgumentParser(
        description="Brain Aging Simulation using WGAN-GP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--filters", type=int, default=32,
                        help="Base number of filters in conv layers")
    parser.add_argument("--latent_space", type=int, default=130,
                        help="Latent space dimension")
    parser.add_argument("--age_dim", type=int, default=100,
                        help="Age vector dimension")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=600,
                        help="Number of training epochs")
    parser.add_argument("--critic_iter", type=int, default=5,
                        help="Critic iterations per generator iteration")
    parser.add_argument("--gp_weight", type=float, default=10.0,
                        help="Gradient penalty weight")
    parser.add_argument("--id_weight", type=float, default=100.0,
                        help="Identity loss weight")
    parser.add_argument("--self_rec_weight", type=float, default=10.0,
                        help="Self-reconstruction loss weight")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA training")

    # Paths
    parser.add_argument("--data_root", type=str, default="./data/",
                        help="Root directory for brain MRI data")
    parser.add_argument("--paired_list_csv", type=str, default="./paired_image_files.csv",
                        help="CSV file with paired young/old image paths")
    parser.add_argument("--model_save_path", type=str, default="./result/model/",
                        help="Directory to save model checkpoints")
    parser.add_argument("--sample_image_save_path", type=str, default="./result/sample_image/",
                        help="Directory to save sample images")

    args = parser.parse_args()

    config = Config(
        lr=args.lr,
        batch_size=args.batch_size,
        filters=args.filters,
        latent_space=args.latent_space,
        age_dim=args.age_dim,
        epochs=args.epochs,
        critic_iter=args.critic_iter,
        gp_weight=args.gp_weight,
        id_weight=args.id_weight,
        self_rec_weight=args.self_rec_weight,
        use_cuda=not args.no_cuda,
        data_root=args.data_root,
        paired_list_csv=args.paired_list_csv,
        model_save_path=args.model_save_path,
        sample_image_save_path=args.sample_image_save_path,
    )

    return config
