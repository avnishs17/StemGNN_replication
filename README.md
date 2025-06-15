# StemGNN: Spectral Temporal Graph Neural Network

A clean and efficient implementation of the Spectral Temporal Graph Neural Network for multivariate time series forecasting, specifically optimized for traffic prediction tasks.

## Overview

This implementation provides a streamlined version of StemGNN that combines:
- **Spectral Convolution**: Efficient FFT-based operations for capturing frequency domain patterns
- **Graph Attention**: Adaptive graph learning for discovering spatial relationships
- **Temporal Modeling**: Multi-scale temporal pattern extraction
- **Memory Efficiency**: Optimized for large-scale datasets

## Architecture

```
Input → Spectral Conv → Graph Attention → Temporal Conv → Output
  ↓         ↓              ↓              ↓         ↓
[B,T,N] → [B,N,D,T] → [B,N,T,D] → [B,N,T,D] → [B,H,N]
```

Where:
- `B`: Batch size
- `T`: Input time steps (window size)
- `N`: Number of nodes (sensors)
- `D`: Hidden dimensions
- `H`: Forecast horizon

## Key Features

### 🚀 Performance Optimizations
- **Efficient FFT Operations**: Spectral convolution using PyTorch's native FFT
- **Multi-head Attention**: Parallel attention computation for graph learning
- **Memory-efficient Training**: Gradient checkpointing and optimized data loading
- **GPU Acceleration**: Full CUDA support with mixed precision training

### 🎯 Model Components
- **GLU Activation**: Gated Linear Units for better gradient flow
- **Spectral Convolution**: Frequency domain processing for temporal patterns
- **Graph Attention**: Learnable adjacency matrix for spatial relationships
- **Residual Connections**: Skip connections for training stability

## Dataset: PeMS07

The model is trained on the PeMS07 traffic dataset:
- **Nodes**: 228 traffic sensors
- **Time Steps**: 12,671 observations (5-minute intervals)
- **Features**: Traffic flow data
- **Task**: Multi-step ahead forecasting (3 steps = 15 minutes)

## Quick Start

### Requirements
```bash
 conda create -p env python=3.10.18
 # make sure you are in folder where the environment was created.
 conda activate ./env             

 pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### Local Training
```bash
python stemgnn_implementation.py
```

### Cloud Training (Modal)
```bash
pip install modal
modal token new
modal run run_on_modal.py
```

## Configuration

Default hyperparameters optimized for PeMS07:

```python
config = {
    'num_nodes': 228,           # Number of traffic sensors
    'window_size': 12,          # Input sequence length (1 hour)
    'horizon': 3,               # Forecast horizon (15 minutes)
    'hidden_dim': 64,           # Model hidden dimensions
    'num_layers': 3,            # Number of StemGNN blocks
    'num_heads': 8,             # Multi-head attention heads
    'batch_size': 32,           # Training batch size
    'learning_rate': 1e-3,      # AdamW learning rate
    'epochs': 100,              # Maximum training epochs
    'early_stop_patience': 15   # Early stopping patience
}
```

## Model Architecture Details

### StemGNN Block
Each StemGNN block contains:

1. **Spectral Convolution Layer**
   - Applies FFT to convert time series to frequency domain
   - Performs convolution in spectral space
   - Applies inverse FFT to return to time domain

2. **Graph Attention Layer**
   - Learns adaptive adjacency matrix
   - Multi-head attention mechanism
   - Captures spatial dependencies between sensors

3. **Temporal Convolution**
   - Depthwise separable convolution
   - Captures local temporal patterns
   - Efficient parameter usage

4. **Gated Linear Unit (GLU)**
   - Applies gating mechanism for selective information flow
   - Improves gradient flow during training

### Data Flow
```python
# Input: [batch_size, window_size, num_nodes]
x = input_data  # [32, 12, 228]

# Project to hidden dimensions
x = input_projection(x)  # [32, 228, 12, 64]

# Apply StemGNN blocks
for block in stem_blocks:
    x, adj_matrix = block(x)  # [32, 228, 12, 64]

# Global pooling and output projection
x = global_pool(x)  # [32, 228, 64]
forecast = output_projection(x)  # [32, 3, 228]
```

## Training Process

### Data Preparation
1. **Normalization**: Z-score normalization using training statistics
2. **Sliding Window**: Create overlapping sequences for training
3. **Train/Val/Test Split**: 70%/15%/15% temporal split

### Training Loop
1. **Forward Pass**: Compute predictions and adjacency matrix
2. **Loss Calculation**: MSE loss between predictions and targets
3. **Backpropagation**: AdamW optimizer with gradient clipping
4. **Validation**: Early stopping based on validation loss
5. **Learning Rate Scheduling**: Cosine annealing schedule

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error

## File Structure

```
├── stemgnn_implementation.py  # Main implementation
├── run_on_modal.py           # Modal deployment script
├── data/
│   └── PeMS07.csv           # Traffic dataset
├── README.md                # This documentation
└── requirements.txt         # Dependencies
```

## Implementation Classes

### Core Components
- `StemGNN`: Main model class
- `StemGNNBlock`: Individual building block
- `SpectralConvLayer`: FFT-based convolution
- `GraphAttention`: Adaptive graph attention
- `GLU`: Gated Linear Unit activation

### Training & Data
- `Trainer`: Training loop management
- `TimeSeriesDataset`: Dataset class for time series data
- `prepare_data()`: Data preprocessing function

## Performance

### Expected Results on PeMS07
Based on the paper and our implementation:
- **MAE**: ~2.14
- **RMSE**: ~4.01  
- **MAPE**: ~5.01%

### Training Time
- **Local (CPU)**: ~2-3 hours
- **Modal (A100)**: ~30-60 minutes

## Advanced Usage

### Custom Configuration
```python
from stemgnn_implementation import StemGNN, Trainer, prepare_data

# Custom model
model = StemGNN(
    num_nodes=228,
    hidden_dim=128,      # Larger model
    num_layers=4,        # Deeper network
    num_heads=16,        # More attention heads
    window_size=24,      # Longer input sequence
    horizon=6            # Longer forecast horizon
)

# Custom training
trainer = Trainer(
    model=model,
    device='cuda',
    learning_rate=5e-4,  # Lower learning rate
    weight_decay=1e-3    # Stronger regularization
)
```

### Different Datasets
To use with other datasets:
1. Ensure CSV format: [time_steps, nodes]
2. Update `num_nodes` in configuration
3. Adjust `window_size` and `horizon` as needed
4. Modify normalization if required

## Modal Deployment

The implementation supports cloud training via Modal:

```python
# run_on_modal.py configuration
image = modal.Image.debian_slim(python_version="3.10")
    .pip_install(["torch==2.0.1+cu118"])
    .pip_install(["pandas", "numpy", "scikit-learn"])
    .add_local_file("stemgnn_implementation.py")
    .add_local_dir("data")

@app.function(gpu="A100-40GB", timeout=7200)
def train_model():
    # Training logic here
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` (try 16 or 8)
   - Reduce `hidden_dim` (try 32 or 48)
   - Use gradient checkpointing

2. **Training Instability**
   - Lower learning rate (try 1e-4)
   - Increase weight decay (try 1e-3)
   - Check data normalization

3. **Poor Performance**
   - Ensure proper train/val/test split
   - Check data preprocessing
   - Verify target scaling

### Memory Requirements
- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB VRAM
- **Optimal**: 32GB RAM, 16GB+ VRAM

## References

1. **Original Paper**: "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting" (NeurIPS 2020)
2. **Dataset**: PeMS (Performance Measurement System) from Caltrans
3. **Framework**: PyTorch 2.0+

## License

This implementation is provided for research and educational purposes. Please cite the original paper if you use this code in your research.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
