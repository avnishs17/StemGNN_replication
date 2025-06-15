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
import numpy as np
from torch.utils.data import DataLoader
import time
import json
import warnings

from model import StemGNN
from trainer import Trainer
from utils import prepare_data

warnings.filterwarnings('ignore')


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
