import modal
import os
import sys

# Define the Modal Image with optimized dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        ["torch==2.0.1+cu118"], extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        [
            "pandas==1.5.3",
            "numpy==1.24.3", 
            "scikit-learn==1.3.0",
        ]
    )    # Copy the optimized StemGNN implementation
    .add_local_file("stemgnn_implementation.py", "/root/stemgnn_implementation.py")
    # Copy the data directory
    .add_local_dir("data", "/root/data")
)

app = modal.App(name="stemgnn-training")

# Use the "output" volume created by the user
output_volume = modal.Volume.from_name("output", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB", 
    timeout=7200,  # 2 hours
    volumes={"/mnt/output": output_volume},
    secrets=[modal.Secret.from_dict({"PYTHONPATH": "/root"})]
)
def train_optimized_stemgnn_on_modal():
    """Train optimized StemGNN model on PeMS07 dataset."""
    print("Modal Job: Starting optimized StemGNN training on Modal...")
    
    log_file_path = "/mnt/output/training_run.log"

    def log_to_volume(message):
        print(message)
        with open(log_file_path, "a") as f:
            f.write(message + "\n")

    log_to_volume("Modal Job: Initializing optimized StemGNN training.")
    log_to_volume(f"Modal Job: Current working directory: {os.getcwd()}")
    log_to_volume(f"Modal Job: Python path: {sys.path}")

    try:
        # Set up directories
        os.makedirs("/mnt/output/models", exist_ok=True)
        
        # Import the optimized implementation
        sys.path.insert(0, '/root')
        from stemgnn_implementation import (
            StemGNN, 
            Trainer, 
            prepare_data
        )
        import torch
        import json
        from datetime import datetime
        
        log_to_volume("Modal Job: Successfully imported optimized StemGNN components.")
        
        # Check GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_to_volume(f"Modal Job: Using device: {device}")
        
        if device.type == 'cuda':
            log_to_volume(f"Modal Job: GPU Name: {torch.cuda.get_device_name()}")
            log_to_volume(f"Modal Job: GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Configuration for PeMS07 
        config = {
            'data_path': '/root/data/PeMS07.csv',
            'num_nodes': 228,
            'window_size': 12,
            'horizon': 3,
            'hidden_dim': 64,
            'num_layers': 3,
            'num_heads': 8,
            'batch_size': 64,  # Larger batch size for A100
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 100,
            'early_stop_patience': 15,
            'device': str(device)
        }
        
        log_to_volume(f"Modal Job: Training configuration: {json.dumps(config, indent=2)}")
        
        # Prepare data
        log_to_volume("Modal Job: Preparing data...")
        train_dataset, val_dataset, test_dataset = prepare_data(
            config['data_path'],
            window_size=config['window_size'],
            horizon=config['horizon']
        )
        
        log_to_volume(f"Modal Job: Train dataset: {len(train_dataset)} samples")
        log_to_volume(f"Modal Job: Val dataset: {len(val_dataset)} samples")
        log_to_volume(f"Modal Job: Test dataset: {len(test_dataset)} samples")
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
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
        log_to_volume("Modal Job: Creating optimized StemGNN model...")
        model = StemGNN(
            num_nodes=config['num_nodes'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            horizon=config['horizon']
        )
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_to_volume(f"Modal Job: Model created with {total_params:,} trainable parameters")
          # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Training phase
        log_to_volume("Modal Job: Starting training phase...")
        start_time = datetime.now().timestamp()
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            early_stop_patience=config['early_stop_patience'],
            save_path='/mnt/output/models/stemgnn_best_model.pt'
        )
        
        training_time = (datetime.now().timestamp() - start_time) / 60
        log_to_volume(f'Modal Job: Training completed in {training_time:.2f} minutes')
          # Test model
        log_to_volume("Modal Job: Starting evaluation phase...")
        
        # Denormalize test results for interpretable metrics
        import numpy as np
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
        
        log_to_volume("Modal Job: Test Results (Denormalized):")
        log_to_volume(f"Modal Job: MAE:  {test_mae:.4f}")
        log_to_volume(f"Modal Job: RMSE: {test_rmse:.4f}")
        log_to_volume(f"Modal Job: MAPE: {test_mape:.2f}%")
        
        # Save results
        results = {
            'config': config,
            'training_time_minutes': training_time,
            'history': {k: [float(x) for x in v] for k, v in history.items()},
            'test_metrics': {
                'mae': float(test_mae),
                'rmse': float(test_rmse),
                'mape': float(test_mape)
            }
        }
        
        with open('/mnt/output/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        log_to_volume("Modal Job: Results saved to /mnt/output/training_results.json")
        log_to_volume("Modal Job: Model saved to /mnt/output/models/stemgnn_best_model.pt")
        
        # Commit volume to persist results
        output_volume.commit()
        
        log_to_volume("Modal Job: Optimized StemGNN training completed successfully!")
        return f"Training completed! MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%"

    except ImportError as e:
        log_to_volume(f"Modal Job: ImportError: {e}")
        log_to_volume("Modal Job: Failed to import optimized StemGNN components. Check file locations.")
        output_volume.commit()
        return f"Modal Job: ImportError - {e}"
    except FileNotFoundError as e:
        log_to_volume(f"Modal Job: FileNotFoundError: {e}")
        log_to_volume("Modal Job: Ensure data files are correctly placed.")
        output_volume.commit()
        return f"Modal Job: FileNotFoundError - {e}"
    except Exception as e:
        import traceback
        error_message = f"Modal Job: An unexpected error occurred: {e}\n{traceback.format_exc()}"
        log_to_volume(error_message)
        output_volume.commit()
        return f"Modal Job: Error during training - {e}"

@app.local_entrypoint()
def main():
    print("Local: Submitting optimized StemGNN training job to Modal...")
    call = train_optimized_stemgnn_on_modal.spawn()
    print(f"Local: Modal job submitted. Call ID: {call.object_id}")
    print("Local: You can track progress on the Modal dashboard or via CLI.")
    print(f"Local: To see live output: modal logs {call.object_id}")
    print("Local: Waiting for remote job to complete...")
    try:
        result = call.get(timeout=7300)  # Wait for the result, timeout slightly longer than function timeout
        print(f"Local: Modal job completed. Result: {result}")
    except Exception as e:
        print(f"Local: Modal job failed or timed out. Error: {e}")
        print(f"Local: Check remote logs with: modal logs {call.object_id}")

if __name__ == "__main__":
    # This is primarily for the `modal run` command to find the local_entrypoint.
    # Direct execution `python run_on_modal.py` will define the app but not necessarily run main correctly for Modal.
    # to run on modal first `pip install modal` then run `modal token new` Sign in with modal.com account and 
    # once its token is successfully authenticated.
    # run in the environment you installed modal. ` modal run run_on_modal.py`
    pass


