# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brain aging simulation using a GAN-based model (WGAN-GP architecture) in PyTorch. The system takes paired young/old brain MRI images and learns to generate age-progressed brain images from younger ones.

## Running the Project

```bash
# Install dependencies (requires PyTorch with CUDA 11.3)
pip install -r requirements.txt

# Train the model
python main.py
```

## Configuration

All hyperparameters and paths are in `configurations/experiment_params.py`:
- `data_root`: Directory containing brain MRI `.npy` files
- `paired_list_csv`: CSV mapping young/old image pairs with ages
- `model_save_path` / `sample_image_save_path`: Output directories
- Training params: `lr`, `batch_size`, `epochs`, `critic_iter`, `gp_weight`, etc.

## Architecture

### Core Components

**Generator** (`models/generator.py`): U-Net style encoder-decoder that generates aging masks
- Encoder: 4 conv blocks with pooling, compresses to latent space (130-dim)
- Bottleneck: Concatenates latent vector with age gap vector (100-dim ordinal encoding)
- Decoder: Upsampling with skip connections from encoder
- Output is a residual mask added to input: `generated = young_image + G(young_image, age_gap)`

**Discriminator** (`models/discriminator.py`): Conditional discriminator that evaluates realism given target age
- Similar encoder structure to Generator
- Takes image + age vector as input
- Uses Global Average Pooling for final output

**Training** (`models/training.py`): WGAN-GP training loop
- Critic trained `critic_iter` (5) times per generator update
- Losses: Wasserstein loss + gradient penalty + identity loss + self-reconstruction loss
- Identity loss: Penalizes changes when age gap is small
- Self-reconstruction loss: When age_gap=0, output should equal input

### Data Pipeline

**Dataset** (`utils/data_loader.py`):
- Loads paired young/old brain images from `.npy` files
- Age encoded as ordinal vector via `age2ord_vector()`: 100-dim vector where trailing N elements are 1s (N = age)
- Images cropped to 208x160

### Loss Functions (`utils/cost.py`)
- `identity_loss`: L1 weighted by inverse age gap (preserve identity for small transformations)
- `self_rec_loss`: L1 between input and zero-gap output

## Key Patterns

- Age information passed as ordinal vectors, not scalars
- Generator produces residual masks (added to input image)
- Uses WGAN-GP formulation with gradient penalty for stable training
- Models saved every 30 epochs to `model_save_path`
