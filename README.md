# Brain Aging Simulation

PyTorch implementation of **"Learning to synthesise the ageing brain without longitudinal data"** (Medical Image Analysis, 2021)

[![Paper](https://img.shields.io/badge/Paper-MedIA-blue)](https://www.sciencedirect.com/science/article/abs/pii/S1361841521002152)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange)](https://pytorch.org/)

## Overview

This project simulates brain aging progression from MRI images using a conditional GAN (WGAN-GP) architecture. Given a young brain MRI and a target age, the model generates a realistic aged brain image while preserving individual anatomical characteristics.

**Key Features:**
- Learns aging patterns from cross-sectional data (no longitudinal data required)
- Age-conditioned generation using ordinal age encoding
- Residual learning for subtle aging transformations
- WGAN-GP for stable training

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Generator (U-Net)                        │
├─────────────────────────────────────────────────────────────────┤
│  Young Brain MRI ──► Encoder ──► Latent + Age Gap ──► Decoder   │
│         │                                                  │     │
│         └──────────── Skip Connections ───────────────────►│     │
│                                                            ▼     │
│                                              Aging Mask (residual)│
│                                                            │     │
│  Output = Young Brain + Aging Mask ◄──────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Discriminator (Conditional)                   │
├─────────────────────────────────────────────────────────────────┤
│  Brain Image + Target Age ──► Encoder ──► Real/Fake Score       │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Functions

| Loss | Description |
|------|-------------|
| **Wasserstein Loss** | Adversarial loss for realistic image generation |
| **Gradient Penalty** | Enforces Lipschitz constraint for stable WGAN training |
| **Identity Loss** | Preserves identity when age gap is small (weighted by inverse age gap) |
| **Self-Reconstruction** | When age_gap=0, output should equal input |

## Installation

```bash
# Clone repository
git clone https://github.com/username/brain-aging-simulation.git
cd brain-aging-simulation

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 1.12+ (with CUDA 11.3 for GPU training)
- See `requirements.txt` for full dependencies

## Usage

### Training

```bash
# Basic training
python main.py --data_root ./data/ --paired_list_csv ./paired_images.csv

# Custom hyperparameters
python main.py \
    --batch_size 16 \
    --epochs 300 \
    --lr 0.0001 \
    --gp_weight 10 \
    --id_weight 100
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./data/` | Directory containing brain MRI .npy files |
| `--paired_list_csv` | `./paired_image_files.csv` | CSV file mapping young/old image pairs |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 600 | Number of training epochs |
| `--lr` | 0.0001 | Learning rate |
| `--no_cuda` | False | Disable CUDA training |

Run `python main.py --help` for full options.

### Data Format

The CSV file should contain paired young/old brain images:

```csv
young_filename,young_age,old_filename,old_age
subject001_young.npy,25,subject001_old.npy,65
subject002_young.npy,30,subject002_old.npy,70
```

Brain MRI data should be preprocessed and saved as `.npy` files with shape `(H, W)`.

## Project Structure

```
brain-aging-simulation/
├── main.py                      # Training entry point
├── configurations/
│   └── config.py                # Configuration (dataclass + argparse)
├── models/
│   ├── generator.py             # U-Net Generator with age conditioning
│   ├── discriminator.py         # Conditional Discriminator
│   ├── layers.py                # Custom layers (Conv2D, Deconv2D, etc.)
│   └── training.py              # WGAN-GP training loop
├── utils/
│   ├── data_loader.py           # Dataset and age encoding
│   └── cost.py                  # Loss functions
├── requirements.txt
└── README.md
```

## Technical Details

- **Age Encoding**: Ordinal vector representation (100-dim) where trailing N elements are 1s (N = age)
- **Image Size**: 208 × 160 (after cropping)
- **Generator Output**: Residual mask added to input image
- **Discriminator Updates**: 5 critic iterations per generator update

## References

```bibtex
@article{xia2021learning,
  title={Learning to synthesise the ageing brain without longitudinal data},
  author={Xia, Tian and Chartsias, Agisilaos and Tsaftaris, Sotirios A and others},
  journal={Medical Image Analysis},
  volume={73},
  pages={102169},
  year={2021},
  publisher={Elsevier}
}
```

## License

This project is for educational and research purposes.
