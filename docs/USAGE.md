# Usage Guide

This document provides comprehensive instructions for using the Conditional Diffusion Models for Medical Imaging (CDM4MI) project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd conditional-dm4mi

# Create and activate conda environment
conda env create -f environment.yaml
conda activate cdm4mi

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### 2. Basic Training

```bash
# Train with default configuration
python run_scripts/dm4mi-conditional.py

# Train with custom config
python run_scripts/dm4mi-conditional.py --config run_scripts/configs/ddpm-cond/exp01_brain_mr_ixi_gaussian_config.yaml
```

### 3. Basic Inference

```python
from models.ddpm_conditioned import ConditionDDPM
from utils.initialize_configs import instantiate_from_configs
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('run_scripts/configs/ddpm-cond/exp01_brain_1BA001_ct_config.yaml')

# Initialize model
model = instantiate_from_configs(config.model)

# Load checkpoint
model.load_state_dict(torch.load('path/to/checkpoint.ckpt'))

# Generate samples
samples = model.sample(batch_size=16)
```

## Data Preparation

### Converting Medical Images to HDF5

The project uses HDF5 format for efficient data loading. Convert your medical images using the provided utility:

```python
from utils.create_training_images import save_as_hdf5

# Convert CT image
save_as_hdf5(
    inputpath="data/brain/1BA001/ct.nii.gz",
    datasetname="ct",
    targetdir="data/brain/1BA001/ct_h5files_64x64_bs_1_s_32",
    image_size=64,
    stride=32,
    batch_size=1
)

# Convert MR image
save_as_hdf5(
    inputpath="data/brain/1BA001/mr.nii.gz",
    datasetname="mr",
    targetdir="data/brain/1BA001/mr_h5files_64x64_bs_1_s_32",
    image_size=64,
    stride=32,
    batch_size=1
)
```

### Data Directory Structure

```
data/
├── brain/
│   ├── 1BA001/
│   │   ├── ct.nii.gz
│   │   ├── mr.nii.gz
│   │   ├── mask.nii.gz
│   │   ├── ct_h5files_64x64_bs_1_s_32/
│   │   └── mr_h5files_64x64_bs_1_s_32/
│   └── 1BA005/
│       └── ...
```

### Supported Data Formats

- **Input**: NIfTI (.nii.gz) files
- **Processing**: HDF5 files for training
- **Output**: TIFF files for visualization

## Training

### 1. Configuration

Create or modify configuration files in `run_scripts/configs/`:

```yaml
# Example configuration
model:
  target: models.ddpm_conditioned.ConditionDDPM
  params:
    dataset:
      target: diffusion_modules.diffusion_utils.dataloader.load_brain_hdf5
      params:
        image_dir: ./data/brain/1BA001/ct_h5files_64x64_bs_1_s_32
        noise_type: gaussian
        variance: 0.01
    
    image_size: 64
    channels: 1
    batch_size: 16
    timesteps: 1000
    learning_rate: 2.0e-6
    loss_type: "l2"
    beta_schedule: "linear"
    parameterization: "eps"
    conditioning_key: "concat"
    
    unet_rosinality_config:
      target: diffusion_modules.unet_arch.unet_rosinality.Unet
      params:
        in_channel: 2
        out_channel: 1
        channel: 64
        channel_multiplier: [1,2,4]
        n_res_blocks: 2
        attn_strides: [4]
        attn_heads: 1
        dropout: 0.1
```

### 2. Training Scripts

#### Using Python Script
```bash
python run_scripts/dm4mi-conditional.py
```

#### Using Jupyter Notebook
```python
# Open and run the notebook
jupyter notebook run_scripts/dm4mi-conditional-mr-ixi-trained.ipynb
```

### 3. Training Parameters

Key training parameters you can adjust:

- **`image_size`**: Input image dimensions (64, 128, 256)
- **`batch_size`**: Training batch size
- **`learning_rate`**: Learning rate (typically 1e-6 to 1e-4)
- **`timesteps`**: Number of diffusion timesteps (1000 recommended)
- **`max_tsteps`**: Maximum training steps
- **`loss_type`**: Loss function ("l1", "l2", "huber")
- **`beta_schedule`**: Noise schedule ("linear", "cosine", "sqrt")

### 4. Monitoring Training

The project uses PyTorch Lightning for training monitoring:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Setup logging
logger = WandbLogger(project="cdm4mi")

# Setup checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss_simple",
    save_top_k=2,
    filename="best-{epoch:02d}-{val_loss:.2f}"
)

# Train
trainer = Trainer(
    max_steps=50000,
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback]
)
trainer.fit(model)
```

## Inference

### 1. Loading Trained Models

```python
import torch
from models.ddpm_conditioned import ConditionDDPM
from utils.initialize_configs import instantiate_from_configs
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('path/to/config.yaml')

# Initialize model
model = instantiate_from_configs(config.model)

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.ckpt')
model.load_state_dict(checkpoint['state_dict'])

# Set to evaluation mode
model.eval()
```

### 2. Generating Samples

#### DDPM Sampling
```python
# Generate samples using DDPM
with torch.no_grad():
    samples = model.sample(
        batch_size=16,
        return_intermediates=False
    )
```

#### DDIM Sampling
```python
from models.ddim import DDIM

# Create DDIM sampler
ddim_sampler = DDIM(model)

# Generate samples using DDIM
samples, intermediates = ddim_sampler.sample(
    S=200,  # Number of DDIM steps
    batch_size=16,
    shape=[1, 64, 64],  # [channels, height, width]
    eta=0.0  # DDIM eta parameter
)
```

### 3. Conditional Generation

```python
# Load conditioning image
cond_image = load_image('path/to/conditioning_image.nii.gz')

# Generate conditioned samples
with torch.no_grad():
    samples = model.sample(
        cond=cond_image,
        batch_size=16
    )
```

## Evaluation

### 1. Image Metrics

```python
from metrics.image_metrics import ImageMetrics

# Initialize metrics
metrics = ImageMetrics()

# Calculate metrics
results = metrics.image_scores(
    ground_truth=gt_images,
    predictions=generated_images,
    mask=roi_mask,
    dynamic_range=2000
)

print(f"PSNR: {results['psnr']:.2f}")
print(f"SSIM: {results['ssim']:.4f}")
print(f"MAE: {results['mae']:.4f}")
```

### 2. Visualizing Results

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(gt, noisy, denoised, num_samples=4):
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Ground truth
        axes[0, i].imshow(gt[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title('Ground Truth')
        axes[0, i].axis('off')
        
        # Noisy input
        axes[1, i].imshow(noisy[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title('Noisy Input')
        axes[1, i].axis('off')
        
        # Denoised output
        axes[2, i].imshow(denoised[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title('Denoised Output')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize results
visualize_results(gt_images, noisy_images, denoised_images)
```

### 3. Saving Results

```python
from utils.fileio import save_as_tiff

# Save generated samples
save_as_tiff(samples, "generated_samples.tiff")

# Save ground truth for comparison
save_as_tiff(gt_images, "ground_truth.tiff")
```

## Configuration

### 1. Model Configuration

Key model parameters:

```yaml
# Model architecture
unet_rosinality_config:
  in_channel: 2          # Input channels (noisy + conditioning)
  out_channel: 1         # Output channels
  channel: 64            # Base channel count
  channel_multiplier: [1,2,4]  # Multi-scale processing
  n_res_blocks: 2        # Residual blocks per level
  attn_strides: [4]      # Attention at 4x resolution
  attn_heads: 1          # Number of attention heads
  dropout: 0.1           # Dropout rate

# Training parameters
timesteps: 1000          # Diffusion timesteps
learning_rate: 2.0e-6    # Learning rate
batch_size: 16           # Batch size
loss_type: "l2"          # Loss function
beta_schedule: "linear"  # Noise schedule
parameterization: "eps"  # Prediction target
conditioning_key: "concat" # Conditioning method
```

### 2. Data Configuration

```yaml
# Dataset configuration
dataset:
  target: diffusion_modules.diffusion_utils.dataloader.load_brain_hdf5
  params:
    image_dir: ./data/brain/1BA001/ct_h5files_64x64_bs_1_s_32
    noise_type: gaussian
    variance: 0.01
    dynamic_range: 255
```

### 3. Training Configuration

```yaml
# Training setup
max_tsteps: 50000        # Maximum training steps
num_of_train_samples: 1000  # Training samples
num_of_val_samples: 200     # Validation samples
use_ema: true            # Exponential moving average
ema_decay_factor: 0.9999 # EMA decay rate
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
batch_size: 8  # Instead of 16

# Reduce image size
image_size: 32  # Instead of 64

# Use gradient accumulation
accumulate_grad_batches: 2
```

#### 2. Training Instability
```yaml
# Reduce learning rate
learning_rate: 1.0e-6  # Instead of 2.0e-6

# Use gradient clipping
gradient_clip_val: 1.0

# Enable warmup
warmup_steps: 1000
```

#### 3. Poor Quality Results
```yaml
# Increase model capacity
channel: 128  # Instead of 64
channel_multiplier: [1,2,4,8]  # More levels

# Adjust noise schedule
beta_schedule: "cosine"  # Instead of "linear"

# Increase training steps
max_tsteps: 100000
```

#### 4. Data Loading Issues
```python
# Check data paths
import os
print(os.path.exists("data/brain/1BA001/ct_h5files_64x64_bs_1_s_32"))

# Verify HDF5 files
import h5py
with h5py.File("data/brain/1BA001/ct_h5files_64x64_bs_1_s_32/ct_0.hdf5", "r") as f:
    print(f['data'].shape)
```

### Debugging Tips

1. **Check Data Loading**: Verify that data is loaded correctly
2. **Monitor Losses**: Track training and validation losses
3. **Visualize Samples**: Check intermediate denoising steps
4. **Validate Conditioning**: Ensure conditioning images are properly aligned
5. **Check Memory Usage**: Monitor GPU memory usage during training

### Performance Optimization

1. **Use Mixed Precision**: Enable automatic mixed precision
2. **Optimize Data Loading**: Use multiple workers and pin memory
3. **Batch Processing**: Use larger batch sizes when possible
4. **Model Parallelism**: Use multiple GPUs for training

```python
# Enable mixed precision
trainer = Trainer(
    precision=16,  # Mixed precision
    accelerator="gpu",
    devices=2,     # Multiple GPUs
    strategy="ddp" # Distributed training
)
```

## Advanced Usage

### Custom Noise Schedules

```python
# Define custom beta schedule
def custom_beta_schedule(timesteps):
    # Your custom implementation
    return betas

# Use in configuration
beta_schedule: "custom"
```

### Custom Loss Functions

```python
# Define custom loss
def custom_loss(pred, target):
    # Your custom implementation
    return loss

# Use in model
model.loss_type = "custom"
```

### Custom Conditioning

```python
# Implement custom conditioning strategy
class CustomConditioning(nn.Module):
    def forward(self, x, cond):
        # Your custom implementation
        return conditioned_x
```
