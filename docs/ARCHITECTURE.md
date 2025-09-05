# Architecture Documentation

## Model Architecture Overview

This document provides detailed information about the model architectures used in the Conditional Diffusion Models for Medical Imaging (CDM4MI) project.

## Core Components

### 1. Conditional DDPM (ConditionDDPM)

The main model class that extends the base DDPM implementation with conditioning capabilities.

**Key Features:**
- Concatenation-based conditioning
- Support for multiple noise schedules
- Configurable loss functions (L1, L2, Huber)
- EMA (Exponential Moving Average) support

**Architecture:**
```python
class ConditionDDPM(DDPM):
    def __init__(self, cond=None, timesteps=1000, ...):
        # Initializes with conditioning support
        self.cond = cond
        # Uses DiffusionWrapper for UNet integration
```

**Conditioning Strategy:**
- **Input**: Noisy image + conditioning image
- **Method**: Concatenation along channel dimension
- **Output**: Denoised image

### 2. UNet Architecture (Rosinality UNet)

The backbone neural network for the diffusion model, based on the Rosinality UNet implementation.

**Architecture Details:**
- **Input Channels**: 2 (noisy image + conditioning image)
- **Output Channels**: 1 (denoised image)
- **Base Channels**: 64 (configurable)
- **Channel Multipliers**: [1, 2, 4] for multi-scale processing
- **Residual Blocks**: 2 per resolution level
- **Attention**: Self-attention at 4x resolution

**Key Components:**

#### ResBlock with Attention
```python
class ResBlockWithAttention(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, 
                 dropout, use_attention=False, attention_head=1):
        # Residual block with optional self-attention
```

#### Time Embedding
```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        # Sinusoidal positional encoding for timesteps
```

#### Self-Attention
```python
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        # Multi-head self-attention mechanism
```

### 3. Diffusion Wrapper

Handles the integration between the UNet and the diffusion process.

**Conditioning Keys:**
- `concat`: Concatenation-based conditioning
- `crossattn`: Cross-attention conditioning
- `adm`: Adaptive layer normalization

**Implementation:**
```python
class DiffusionWrapper(pl.LightningModule):
    def forward(self, x, t, cond=None):
        if self.conditioning_key == 'concat':
            xc = torch.concat((x, cond), dim=1)
            out = self.diffusion_model(xc, t)
```

## Noise Schedules

### Beta Schedules
The project supports multiple noise schedules:

1. **Linear Schedule**: `β_t = β_1 + (β_T - β_1) * t/T`
2. **Cosine Schedule**: `β_t = f(t/T)` where f is cosine function
3. **Sqrt Schedule**: Square root based scheduling
4. **Custom Schedules**: User-defined beta values

### Parameterization
Two parameterization methods are supported:

1. **Epsilon (ε)**: Predicts the noise added to the image
2. **x0**: Predicts the original clean image

## Sampling Methods

### 1. DDPM Sampling
Standard denoising diffusion probabilistic model sampling:
- **Steps**: 1000 (configurable)
- **Process**: Iterative denoising from pure noise
- **Quality**: High quality, slower inference

### 2. DDIM Sampling
Denoising Diffusion Implicit Models for faster sampling:
- **Steps**: 200 (configurable)
- **Process**: Deterministic sampling with fewer steps
- **Quality**: Good quality, faster inference

## Data Flow

### Training Process
1. **Input**: Clean image + Noisy image pair
2. **Noise Addition**: Random timestep noise injection
3. **Forward Pass**: UNet predicts noise/clean image
4. **Loss Calculation**: MSE/Huber loss between prediction and target
5. **Backpropagation**: Gradient updates

### Inference Process
1. **Input**: Noisy image + Conditioning image
2. **Sampling**: DDPM/DDIM sampling process
3. **Output**: Denoised image

## Configuration Parameters

### Model Configuration
```yaml
unet_rosinality_config:
  in_channel: 2          # Input channels (noisy + conditioning)
  out_channel: 1         # Output channels (denoised)
  channel: 64            # Base channel count
  channel_multiplier: [1,2,4]  # Multi-scale channels
  n_res_blocks: 2        # Residual blocks per level
  attn_strides: [4]      # Attention at 4x resolution
  attn_heads: 1          # Attention heads
  use_affine_time: False # Affine time embedding
  dropout: 0.1           # Dropout rate
  fold: 1                # Spatial folding factor
```

### Training Configuration
```yaml
timesteps: 1000          # Diffusion timesteps
max_tsteps: 1e05         # Maximum training steps
learning_rate: 2.0e-6    # Learning rate
batch_size: 16           # Batch size
loss_type: "l2"          # Loss function
beta_schedule: "linear"  # Noise schedule
parameterization: "eps"  # Prediction target
conditioning_key: "concat" # Conditioning method
```

## Memory and Computational Considerations

### Memory Usage
- **Training**: ~8GB GPU memory for 64x64 images, batch size 16
- **Inference**: ~4GB GPU memory for 64x64 images, batch size 16
- **Scaling**: Memory usage scales quadratically with image size

### Computational Complexity
- **Training**: O(T × N × H × W) where T=timesteps, N=batch_size, H×W=image_size
- **Inference**: O(S × N × H × W) where S=sampling_steps
- **DDIM**: ~5x faster than DDPM for inference

## Performance Optimization

### Training Optimizations
1. **Mixed Precision**: Automatic mixed precision training
2. **Gradient Accumulation**: For larger effective batch sizes
3. **EMA**: Exponential moving average for stable training
4. **Learning Rate Scheduling**: Warmup and decay strategies

### Inference Optimizations
1. **DDIM Sampling**: Faster sampling with fewer steps
2. **Batch Processing**: Efficient batch inference
3. **Memory Management**: Gradient checkpointing for large models

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Training Instability**: Check learning rate and gradient clipping
3. **Poor Quality**: Verify noise schedule and conditioning strategy
4. **Slow Training**: Enable mixed precision and optimize data loading

### Debugging Tips
1. **Visualize Samples**: Check intermediate denoising steps
2. **Monitor Losses**: Track training and validation losses
3. **Validate Conditioning**: Ensure conditioning images are properly aligned
4. **Check Data**: Verify data preprocessing and normalization
