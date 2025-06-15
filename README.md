# StemGNN: Spectral Temporal Graph Neural Network

A clean and efficient implementation of the Spectral Temporal Graph Neural Network for multivariate time series forecasting, specifically optimized for traffic prediction tasks.

## Overview

This implementation provides a streamlined version of StemGNN that combines:
- **Spectral Convolution**: Efficient FFT-based operations for capturing frequency domain patterns
- **Graph Attention**: Adaptive graph learning for discovering spatial relationships
- **Temporal Modeling**: Multi-scale temporal pattern extraction
- **Memory Efficiency**: Optimized for large-scale datasets

## Project Structure

```
├── model.py                 # Contains the StemGNN model architecture (StemGNN, StemGNNBlock, SpectralConvLayer, GraphAttention, GLU)
├── dataset.py               # Defines the TimeSeriesDataset class for data handling
├── trainer.py               # Implements the Trainer class for model training and validation
├── utils.py                 # Includes utility functions like prepare_data for data loading and preprocessing
├── stemgnn_implementation.py  # Main script to run local training and evaluation
├── run_on_modal.py           # Script for deploying and running training on Modal
├── data/
│   └── PeMS07.csv           # Traffic dataset (example)
├── README.md                # This documentation
└── requirements.txt         # Python dependencies
```

## Code Modules

### `model.py`
This module defines the core neural network architecture.
- **`GLU(input_dim, output_dim)`**: Gated Linear Unit.
- **`SpectralConvLayer(time_steps, channels, kernel_size)`**: Spectral Convolution Layer using FFT.
- **`GraphAttention(feature_dim, num_heads, dropout)`**: Adaptive graph attention mechanism.
- **`StemGNNBlock(time_steps, feature_dim, num_heads, dropout)`**: A single block of StemGNN, combining spectral and temporal modeling.
- **`StemGNN(num_nodes, input_dim, hidden_dim, output_dim, num_layers, num_heads, window_size, horizon, dropout)`**: The main StemGNN model.

### `dataset.py`
Handles data loading and preparation for PyTorch.
- **`TimeSeriesDataset(data, window_size, horizon, normalize)`**: PyTorch Dataset class for creating input-output samples from time series data. Handles normalization and denormalization.

### `trainer.py`
Manages the training and evaluation process.
- **`Trainer(model, device, learning_rate, weight_decay)`**: Encapsulates the training loop, validation, optimizer, scheduler, and loss calculation.
    - `train_epoch(train_loader)`: Trains the model for one epoch.
    - `validate(val_loader)`: Evaluates the model on the validation set.
    - `train(train_loader, val_loader, epochs, early_stop_patience, save_path)`: Runs the full training process with early stopping.

### `utils.py`
Contains helper functions for data processing.
- **`prepare_data(data_path, train_ratio, val_ratio, window_size, horizon)`**: Loads data from a CSV file, splits it into training, validation, and test sets, and creates `TimeSeriesDataset` instances.

### `stemgnn_implementation.py`
This is the main script for running the model locally.
- It sets up the configuration (hyperparameters, data paths).
- Initializes the model, datasets, data loaders, and trainer.
- Runs the training and evaluation process.
- Saves the best model and training results.

### `run_on_modal.py`
This script facilitates training the model in the cloud using Modal.
- Defines a Modal image with necessary dependencies.
- Sets up a Modal function (`train_optimized_stemgnn_on_modal`) to run the training job on a GPU-accelerated instance.
- Copies project files (`model.py`, `dataset.py`, `trainer.py`, `utils.py`, and data) to the Modal environment.
- Manages logging and saving results to a Modal Volume.

## Environment Setup

1.  **Create a Conda Environment (Recommended)**:
    ```bash
    conda create -p env python=3.10
    conda activate ./env
    ```
    (Ensure you are in the project's root directory when activating if using a local path for the environment).

2.  **Install Dependencies**:
    Install the required Python packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    If you have a CUDA-enabled GPU and want to use it with PyTorch, ensure you install the correct PyTorch version. The `requirements.txt` might specify a CPU version or a generic one. For CUDA, you might need to install it separately or modify `requirements.txt` e.g.:
    ```bash
    # For CUDA 11.8 (example)
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
    pip install pandas numpy scikit-learn
    ```
    Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct command based on your CUDA version.

## Running the Code

### Local Training

To train the model on your local machine:

```bash
python stemgnn_implementation.py
```
This script will:
1. Load data from `data/PeMS07.csv` (or as specified in the script's config).
2. Preprocess the data.
3. Initialize and train the StemGNN model.
4. Evaluate the model on the test set.
5. Save the best model checkpoint (e.g., `stemgnn_best_model.pt`) and training results (e.g., `training_results.json`).

### Cloud Training (Using Modal)

Modal allows you to run your training script in a cloud environment, which is useful for accessing powerful GPUs and managing dependencies easily.

1.  **Install Modal Client**:
    ```bash
    pip install modal
    ```

2.  **Set up Modal Token**:
    You'll need to authenticate with Modal. This usually involves creating an account on the Modal website and then running:
    ```bash
    modal token new
    ```
    Follow the CLI instructions.

3.  **Run the Modal App**:
    Execute the `run_on_modal.py` script using the Modal CLI:
    ```bash
    modal run run_on_modal.py
    ```
    This command will:
    -   Upload your code (`model.py`, `dataset.py`, `trainer.py`, `utils.py`, `run_on_modal.py`) and the `data` directory to Modal.
    -   Build a container image with the specified Python version and dependencies.
    -   Execute the `train_optimized_stemgnn_on_modal` function on a Modal instance (potentially with a GPU, as configured in the script).
    -   Stream logs to your local terminal.
    -   Save outputs (model checkpoints, results JSON, logs) to a Modal Volume (named "output" in the example script).

## Configuration

Key hyperparameters and settings can be adjusted within `stemgnn_implementation.py` (for local runs) or `run_on_modal.py` (for Modal runs). These typically include:

```python
config = {
    'data_path': 'data/PeMS07.csv', # Path to the dataset
    'num_nodes': 228,           # Number of nodes/sensors in the dataset
    'window_size': 12,          # Number of past time steps to use as input
    'horizon': 3,               # Number of future time steps to predict
    'hidden_dim': 64,           # Hidden dimension size in the model
    'num_layers': 3,            # Number of StemGNN blocks
    'num_heads': 8,             # Number of attention heads in GraphAttention
    'batch_size': 32,           # Batch size for training
    'learning_rate': 1e-3,      # Initial learning rate
    'weight_decay': 1e-4,       # Weight decay for optimizer
    'epochs': 100,              # Maximum number of training epochs
    'early_stop_patience': 15,  # Patience for early stopping
    'device': 'cuda' if torch.cuda.is_available() else 'cpu' # Device to train on
}
```
When running on Modal, the `data_path` inside the Modal function will refer to the path within the Modal container (e.g., `/root/data/PeMS07.csv`).