import numpy as np
import torch
from torch.utils.data import Dataset

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
