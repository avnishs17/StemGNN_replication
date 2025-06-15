import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from typing import Dict

from model import StemGNN # Assuming model.py contains StemGNN

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
