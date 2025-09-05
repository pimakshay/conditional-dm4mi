# Conditional Diffusion Models for Medical Imaging (CDM4MI)

A research project exploring conditional Denoising Diffusion Probabilistic Models (DDPM) for medical image denoising, developed as a collaboration between GE Healthcare and Technical University of Munich (TUM).

## Overview

This project investigates the effectiveness of diffusion models for medical image denoising, specifically focusing on:
- **Conditional DDPM** for image denoising tasks
- **Comparison** with traditional methods (NLM, U-Net)
- **Noise level analysis** and performance evaluation
- **Conditioning strategies** and guidance mechanisms

The research aims to understand how diffusion models work in medical imaging contexts and determine optimal combinations of conditioning and guidance for different noise levels.

## Key Features

- **Conditional DDPM Implementation**: Custom implementation with concatenation-based conditioning
- **Multiple Sampling Methods**: Both DDPM and DDIM sampling strategies
- **Medical Image Support**: CT and MR image processing with HDF5 data format
- **Comprehensive Evaluation**: PSNR, SSIM, and MAE metrics
- **Flexible Architecture**: Configurable UNet with attention mechanisms
- **Noise Analysis**: Gaussian noise injection with configurable variance

## Project Structure

```
├── models/                    # Model implementations
│   ├── ddpm_conditioned.py   # Conditional DDPM model
│   ├── ddpm_st_diffusion.py  # Base DDPM implementation
│   └── ddim.py              # DDIM sampling
├── diffusion_modules/        # Core diffusion components
│   ├── unet_arch/           # UNet architectures
│   └── diffusion_utils/     # Data loaders and utilities
├── run_scripts/             # Training and evaluation scripts
│   ├── configs/            # Configuration files
│   └── data/               # Sample data
├── metrics/                # Evaluation metrics
├── utils/                  # Utility functions
└── docs/                  # Documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd conditional-dm4mi
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate cdm4mi
   ```

3. **Install additional dependencies** (if needed):
   ```bash
   pip install wandb  # for experiment tracking
   ```

## Quick Start

### Training a Model

1. **Prepare your data** in HDF5 format (see [Data Preparation](#data-preparation))

2. **Configure your experiment** by modifying the config files in `run_scripts/configs/`

3. **Run training**:
   ```bash
   python run_scripts/dm4mi-conditional.py
   ```

### Using Pre-trained Models

```python
from models.ddpm_conditioned import ConditionDDPM
from utils.initialize_configs import instantiate_from_configs
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('run_scripts/configs/ddpm-cond/exp01_brain_1BA001_ct_config.yaml')

# Initialize model
model = instantiate_from_configs(config.model)
model.load_state_dict(torch.load('path/to/checkpoint.ckpt'))

# Generate samples
samples = model.sample(batch_size=16)
```

## Configuration

The project uses YAML configuration files for easy experimentation. Key parameters include:

- **Model Architecture**: UNet configuration, attention settings
- **Training**: Learning rate, batch size, timesteps
- **Data**: Image size, noise variance, dataset paths
- **Sampling**: DDIM steps, guidance scale

Example configuration:
```yaml
model:
  target: models.ddpm_conditioned.ConditionDDPM
  params:
    image_size: 64
    channels: 1
    timesteps: 1000
    learning_rate: 2.0e-6
    conditioning_key: "concat"
    unet_rosinality_config:
      channel: 64
      channel_multiplier: [1,2,4]
      n_res_blocks: 2
```

## Data Preparation

### Converting Medical Images to HDF5

Use the provided utility to convert NIfTI files to HDF5 format:

```python
from utils.create_training_images import save_as_hdf5

save_as_hdf5(
    inputpath="path/to/image.nii.gz",
    datasetname="ct",
    targetdir="output/hdf5_files",
    image_size=64,
    stride=32
)
```

### Supported Data Formats

- **Input**: NIfTI (.nii.gz) files
- **Processing**: HDF5 format for efficient loading
- **Output**: TIFF files for visualization

## Model Architectures

### Conditional DDPM
- **Conditioning**: Concatenation-based conditioning
- **Noise Schedule**: Linear, cosine, or custom beta schedules
- **Loss Functions**: L1, L2, or Huber loss
- **Parameterization**: Epsilon or x0 prediction

### UNet Architecture
- **Base Channels**: 64 (configurable)
- **Multi-scale**: [1, 2, 4] channel multipliers
- **Attention**: Self-attention at specific resolutions
- **Time Embedding**: Sinusoidal positional encoding

## Evaluation Metrics

The project includes comprehensive evaluation metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MAE**: Mean Absolute Error

```python
from metrics.image_metrics import ImageMetrics

metrics = ImageMetrics()
results = metrics.image_scores(ground_truth, predictions, mask, dynamic_range)
```

## Research Context

This project was conducted as part of a research collaboration between GE Healthcare and TUM, focusing on:

1. **Understanding diffusion model behavior** in medical imaging
2. **Comparing conditioning strategies** for denoising tasks
3. **Evaluating performance** across different noise levels
4. **Benchmarking against traditional methods** (NLM, U-Net)

## Key Findings

- Conditional DDPMs show promising results for medical image denoising
- Concatenation-based conditioning provides effective guidance
- Performance varies significantly with noise levels
- DDIM sampling offers faster inference compared to DDPM


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is a research project. For questions or collaboration, please contact the research team.

## Acknowledgments

- GE Healthcare for research collaboration
- Technical University of Munich for academic support
- PyTorch Lightning team for the training framework
- The open-source medical imaging community