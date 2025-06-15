import pandas as pd
import numpy as np
from typing import Tuple
from dataset import TimeSeriesDataset # Assuming dataset.py contains TimeSeriesDataset

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
