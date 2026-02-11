"""
Brain Aging Simulation - Main Training Script

PyTorch implementation of "Learning to synthesise the ageing brain without longitudinal data"
Paper: https://arxiv.org/abs/1912.02620
"""

import torch
from configurations.config import parse_args
from utils.data_loader import Dataset
from models.training import Synthesized_model


def main():
    # Parse command line arguments
    config = parse_args()

    # Print configuration
    print("=" * 50)
    print("Brain Aging Simulation - Training")
    print("=" * 50)
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"CUDA enabled: {config.use_cuda}")
    print(f"Data root: {config.data_root}")
    print(f"Paired list CSV: {config.paired_list_csv}")
    print("=" * 50)

    # Setup dataset and dataloader
    dataset = Dataset(config.paired_list_csv, config.data_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Initialize and train model
    model = Synthesized_model(config)
    model.train(dataloader, config.epochs)


if __name__ == "__main__":
    main()
