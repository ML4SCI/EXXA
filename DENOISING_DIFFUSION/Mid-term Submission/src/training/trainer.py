#!/usr/bin/env python
"""
Training loop for diffusion-based denoising model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
from datetime import datetime
import numpy as np

from src.models.unet import UNet
from src.models.noise_scheduler import NoiseScheduler


class Trainer:
    """
    Trainer for diffusion model.
    
    Handles training, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: UNet,
        scheduler: NoiseScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "experiments/checkpoints",
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Loss function (MSE between predicted and actual noise)
        self.loss_fn = nn.MSELoss()
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_one_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, img_ids) in enumerate(self.train_loader):
            # Extract channels: channel 0 = dirty, channel 1 = clean
            # For denoising, input is dirty (noisy observation), target is clean
            dirty = x[:, 0:1, :, :].to(self.device)  # (B, 1, H, W)
            clean = x[:, 1:2, :, :].to(self.device)  # (B, 1, H, W)
            
            # Sample random timesteps
            batch_size = dirty.shape[0]
            t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=self.device)
            
            # Forward diffusion: add noise to clean image
            noisy_clean, noise = self.scheduler.q_sample(clean, t)
            
            # Model predicts noise given (noisy_clean, t)
            # But we want to denoise the dirty observation
            # So we condition on dirty as well
            # Concatenate dirty and noisy_clean as input
            x_t = torch.cat([dirty, noisy_clean], dim=1)  # (B, 2, H, W)
            
            # Predict noise
            predicted_noise = self.model(x_t, t)  # (B, 1, H, W)
            
            # Compute loss: MSE between predicted and actual noise
            loss = self.loss_fn(predicted_noise, noise)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: Loss {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set. Returns average loss."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for x, img_ids in self.val_loader:
            # Extract channels
            dirty = x[:, 0:1, :, :].to(self.device)
            clean = x[:, 1:2, :, :].to(self.device)
            
            # Sample random timesteps
            batch_size = dirty.shape[0]
            t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=self.device)
            
            # Forward diffusion
            noisy_clean, noise = self.scheduler.q_sample(clean, t)
            
            # Concatenate input
            x_t = torch.cat([dirty, noisy_clean], dim=1)
            
            # Predict noise
            predicted_noise = self.model(x_t, t)
            
            # Compute loss
            loss = self.loss_fn(predicted_noise, noise)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int = 10) -> Dict:
        """Train the model."""
        print("\n" + "=" * 60)
        print("Training Diffusion Model")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
            
            # Train
            train_loss = self.train_one_epoch()
            self.train_losses.append(train_loss)
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  ✓ Best model saved!")
            else:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model from checkpoint. Returns epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        return checkpoint.get("epoch", 0)


def train_ddpm(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[UNet, Dict]:
    """
    Train a DDPM model for denoising.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        (trained_model, training_results) tuple
    """
    # Create model and scheduler
    model = UNet(
        in_channels=2,
        out_channels=1,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
        time_emb_dim=128,
        num_res_blocks=2,
    )
    
    scheduler = NoiseScheduler(
        timesteps=1000,
        beta_schedule="cosine",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Train
    results = trainer.train(num_epochs=num_epochs)
    
    return model, results
