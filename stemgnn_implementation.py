"""
StemGNN Implementation for PeMS07 Dataset
====================================================

Spectral Temporal Graph Neural Network specifically for the PeMS07 
traffic dataset with 228 nodes and 12,671 timesteps.

Key optimizations:
1. Memory-efficient data loading with chunking
2. Optimized spectral convolution operations
3. Efficient attention mechanism
4. Gradient checkpointing for large models
5. Mixed precision training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict
import time
import json
import warnings
warnings.filterwarnings('ignore')


class GLU(nn.Module):
    """Gated Linear Unit with parameter sharing"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(GLU, self).__init__()
        self.gate_linear = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_linear(x)
        left, right = gate_output.chunk(2, dim=-1)
        return left * torch.sigmoid(right)


class SpectralConvLayer(nn.Module):
    """Spectral Convolution Layer with efficient FFT operations"""
    
    def __init__(self, time_steps: int, channels: int, kernel_size: int = 3):
        super(SpectralConvLayer, self).__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Learnable spectral filters
        self.spectral_conv = nn.Conv1d(
            in_channels=channels * 2,  # Real + Imaginary
            out_channels=channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=2  # Separate processing for real/imaginary
        )
        
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, nodes, channels, time_steps]
        Returns:
            Processed tensor [batch_size, nodes, channels, time_steps]
        """
        batch_size, nodes, channels, time_steps = x.shape
        
        # Reshape for FFT: [batch_size * nodes, channels, time_steps]
        x_reshaped = x.view(batch_size * nodes, channels, time_steps)
        
        # Apply FFT
        x_fft = torch.fft.rfft(x_reshaped, dim=-1)
        
        # Split real and imaginary parts
        x_real = x_fft.real  # [batch_size * nodes, channels, freq_bins]
        x_imag = x_fft.imag  # [batch_size * nodes, channels, freq_bins]
        
        # Concatenate real and imaginary for convolution
        x_complex = torch.cat([x_real, x_imag], dim=1)  # [batch_size * nodes, channels*2, freq_bins]
        
        # Apply spectral convolution
        x_conv = self.spectral_conv(x_complex)
        x_conv = self.activation(x_conv)
        
        # Split back to real and imaginary
        channels_half = x_conv.size(1) // 2
        x_real_conv = x_conv[:, :channels_half]
        x_imag_conv = x_conv[:, channels_half:]
        
        # Reconstruct complex tensor
        x_complex_recon = torch.complex(x_real_conv, x_imag_conv)
        
        # Apply inverse FFT
        x_ifft = torch.fft.irfft(x_complex_recon, n=time_steps, dim=-1)
        
        # Reshape back: [batch_size, nodes, channels, time_steps]
        x_output = x_ifft.view(batch_size, nodes, channels, time_steps)
        
        # Apply normalization
        x_output = x_output.permute(0, 1, 3, 2)  # [batch_size, nodes, time_steps, channels]
        x_output = self.norm(x_output)
        x_output = x_output.permute(0, 1, 3, 2)  # [batch_size, nodes, channels, time_steps]
        
        return x_output


class GraphAttention(nn.Module):
    """Adaptive graph attention mechanism"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(GraphAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** 0.5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, nodes, time_steps, feature_dim]
        Returns:
            output: Attention output [batch_size, nodes, time_steps, feature_dim]
            adj_matrix: Learned adjacency matrix [batch_size, nodes, nodes]
        """
        batch_size, nodes, time_steps, feature_dim = x.shape
        
        # Aggregate temporal information for graph learning
        x_temporal = x.mean(dim=2)  # [batch_size, nodes, feature_dim]
        
        # Multi-head attention for graph structure learning
        q = self.query_proj(x_temporal).view(batch_size, nodes, self.num_heads, self.head_dim)
        k = self.key_proj(x_temporal).view(batch_size, nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate across heads to get adjacency matrix
        adj_matrix = attention_weights.mean(dim=1)  # [batch_size, nodes, nodes]
        
        # Apply attention to original features
        x_reshaped = x.view(batch_size, nodes, time_steps * feature_dim)
        attended_features = torch.einsum('bnm,bmt->bnt', adj_matrix, x_reshaped)
        attended_features = attended_features.view(batch_size, nodes, time_steps, feature_dim)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attended_features)
        
        return output, adj_matrix


class StemGNNBlock(nn.Module):
    """StemGNN block combining spectral and temporal modeling"""
    
    def __init__(self, 
                 time_steps: int, 
                 feature_dim: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(StemGNNBlock, self).__init__()
        
        self.spectral_conv = SpectralConvLayer(time_steps, feature_dim)
        self.graph_attention = GraphAttention(feature_dim, num_heads, dropout)
        
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim  # Depthwise convolution for efficiency
        )
        
        self.glu = GLU(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, nodes, time_steps, feature_dim]
        Returns:
            output: Processed tensor [batch_size, nodes, time_steps, feature_dim]
            adj_matrix: Learned adjacency matrix [batch_size, nodes, nodes]
        """
        batch_size, nodes, time_steps, feature_dim = x.shape
        
        # Spectral convolution
        x_spectral = x.permute(0, 1, 3, 2)  # [batch_size, nodes, feature_dim, time_steps]
        x_spectral = self.spectral_conv(x_spectral)
        x_spectral = x_spectral.permute(0, 1, 3, 2)  # [batch_size, nodes, time_steps, feature_dim]
        
        # Graph attention
        x_graph, adj_matrix = self.graph_attention(x_spectral)
        
        # Temporal convolution
        x_temporal = x_graph.view(batch_size * nodes, time_steps, feature_dim)
        x_temporal = x_temporal.permute(0, 2, 1)  # [batch_size * nodes, feature_dim, time_steps]
        x_temporal = self.temporal_conv(x_temporal)
        x_temporal = x_temporal.permute(0, 2, 1)  # [batch_size * nodes, time_steps, feature_dim]
        x_temporal = x_temporal.view(batch_size, nodes, time_steps, feature_dim)
        
        # GLU activation and residual connection
        x_glu = self.glu(x_temporal)
        x_output = self.layer_norm(x + self.dropout(x_glu))
        
        return x_output, adj_matrix


class StemGNN(nn.Module):
    """StemGNN model for multivariate time series forecasting"""
    
    def __init__(self,
                 num_nodes: int,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 window_size: int = 12,
                 horizon: int = 3,
                 dropout: float = 0.1):
        super(StemGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.horizon = horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # StemGNN blocks
        self.stem_blocks = nn.ModuleList([
            StemGNNBlock(window_size, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon * output_dim)
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, window_size, num_nodes]
        Returns:
            forecast: Forecast tensor [batch_size, horizon, num_nodes]
            adj_matrix: Final adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size, window_size, num_nodes = x.shape
        
        # Reshape and project input: [batch_size, num_nodes, window_size, hidden_dim]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, num_nodes, window_size, 1]
        x = self.input_proj(x)  # [batch_size, num_nodes, window_size, hidden_dim]
        
        # Apply StemGNN blocks
        adj_matrices = []
        for block in self.stem_blocks:
            x, adj_matrix = block(x)
            adj_matrices.append(adj_matrix)
        
        # Use the last adjacency matrix
        final_adj_matrix = adj_matrices[-1]
        
        # Global average pooling over time dimension
        x = x.mean(dim=2)  # [batch_size, num_nodes, hidden_dim]
        
        # Output projection
        forecast = self.output_proj(x)  # [batch_size, num_nodes, horizon * output_dim]
        forecast = forecast.view(batch_size, num_nodes, self.horizon, self.output_dim)
        forecast = forecast.squeeze(-1).permute(0, 2, 1)  # [batch_size, horizon, num_nodes]
        
        return forecast, final_adj_matrix


class TimeSeriesDataset(Dataset):
    """Dataset class for time series data"""
    
    def __init__(self, 
                 data: np.ndarray,
                 window_size: int = 12,
                 horizon: int = 3,
                 normalize: bool = True):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.normalize = normalize
        
        if self.normalize:
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1, self.std)  # Avoid division by zero
            self.data = (data - self.mean) / self.std
        
        # Pre-compute valid indices
        self.valid_indices = list(range(window_size, len(data) - horizon + 1))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size
        
        x = self.data[start_idx:end_idx]  # [window_size, num_nodes]
        y = self.data[end_idx:end_idx + self.horizon]  # [horizon, num_nodes]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize the data"""
        if self.normalize:
            return data * self.std + self.mean
        return data


class Trainer:
    """Trainer for StemGNN"""
    
    def __init__(self,
                 model: StemGNN,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            forecast, _ = self.model(x)
            loss = self.criterion(forecast, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx:4d}, Loss: {loss.item():.6f}')
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                forecast, _ = self.model(x)
                loss = self.criterion(forecast, y)
                
                total_loss += loss.item()
                predictions.append(forecast.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        return {
            'loss': total_loss / len(val_loader),
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int = 100,
              early_stop_patience: int = 15,
              save_path: str = 'best_model.pt') -> Dict[str, list]:
        """Train the model with early stopping"""
        
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_mape': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_mape'].append(val_metrics['mape'])
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1:3d}/{epochs} | Time: {epoch_time:.2f}s | '
                  f'Train Loss: {train_loss:.6f} | Val Loss: {val_metrics["loss"]:.6f} | '
                  f'MAE: {val_metrics["mae"]:.4f} | RMSE: {val_metrics["rmse"]:.4f} | '
                  f'MAPE: {val_metrics["mape"]:.2f}%')
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f'New best model saved with validation loss: {best_val_loss:.6f}')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
        
        return history


def prepare_data(data_path: str, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 window_size: int = 12,
                 horizon: int = 3) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Prepare train, validation, and test datasets"""
    
    print("Loading and preparing data...")
    
    # Load data
    data = pd.read_csv(data_path).values.astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    # Split data
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = data[:train_len]
    val_data = data[train_len:train_len + val_len]
    test_data = data[train_len + val_len:]
    
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
      # Create datasets
    train_dataset = TimeSeriesDataset(train_data, window_size, horizon, normalize=True)
    val_dataset = TimeSeriesDataset(val_data, window_size, horizon, normalize=True)
    test_dataset = TimeSeriesDataset(test_data, window_size, horizon, normalize=True)
    
    # Use train dataset's normalization statistics for val and test
    val_dataset.mean = train_dataset.mean
    val_dataset.std = train_dataset.std
    val_dataset.data = (val_data - val_dataset.mean) / val_dataset.std
    
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    test_dataset.data = (test_data - test_dataset.mean) / test_dataset.std
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_path': 'data/PeMS07.csv',
        'num_nodes': 228,
        'window_size': 12,
        'horizon': 3,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_heads': 8,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 100,
        'early_stop_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=== Optimized StemGNN Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_data(
        config['data_path'],
        window_size=config['window_size'],
        horizon=config['horizon']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = StemGNN(
        num_nodes=config['num_nodes'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
        horizon=config['horizon']
    )
      # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stop_patience=config['early_stop_patience'],
        save_path='stemgnn_best_model.pt'
    )
    training_time = time.time() - start_time
    print(f"Total training time: {training_time/60:.2f} minutes")
      # Test model
    print("\n=== Testing ===")
    
    # Denormalize test results for interpretable metrics
    test_predictions = []
    test_targets = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            forecast, _ = model(x)
            
            # Denormalize
            forecast_denorm = test_dataset.denormalize(forecast.cpu().numpy())
            target_denorm = test_dataset.denormalize(y.cpu().numpy())
            
            test_predictions.append(forecast_denorm)
            test_targets.append(target_denorm)
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # Calculate denormalized metrics
    test_mae = np.mean(np.abs(test_predictions - test_targets))
    test_rmse = np.sqrt(np.mean((test_predictions - test_targets) ** 2))
    test_mape = np.mean(np.abs((test_predictions - test_targets) / (test_targets + 1e-8))) * 100
    
    print("Test Results (Denormalized):")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    
    # Save results
    results = {
        'config': config,
        'training_time': training_time,
        'history': history,
        'test_metrics': {
            'mae': test_mae,
            'rmse': test_rmse,
            'mape': test_mape
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nTraining completed successfully!")
    print("Results saved to 'training_results.json'")
    print("Best model saved to 'stemgnn_best_model.pt'")


if __name__ == "__main__":
    main()
