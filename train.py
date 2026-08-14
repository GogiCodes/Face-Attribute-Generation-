"""
Training script for Conv-VAE with β-schedule and improved optimization.
Includes checkpointing, validation, and TensorBoard logging.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Optional, Tuple
import json

from model import ConvVAE, vae_loss
from config import Config, DEFAULT_CONFIG
from data_loader import get_dataloaders


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for Conv-VAE with advanced features."""
    
    def __init__(self, config: Config, model: ConvVAE):
        self.config = config
        self.model = model
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'runs') if config.training.enable_tensorboard else None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'beta': []
        }
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def get_beta(self, epoch: int) -> float:
        """Compute β value based on schedule."""
        total_epochs = self.config.training.num_epochs
        beta_start = self.config.training.beta_start
        beta_end = self.config.training.beta_end
        warmup_epochs = self.config.training.beta_warmup_epochs
        schedule = self.config.training.beta_schedule
        
        if schedule == 'constant':
            return beta_end
        
        elif schedule == 'linear':
            if epoch < warmup_epochs:
                return beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return beta_start + (beta_end - beta_start) * progress
        
        elif schedule == 'cyclical':
            cycle_length = total_epochs // 3
            progress = (epoch % cycle_length) / cycle_length
            if epoch // cycle_length % 2 == 0:
                return beta_start + (beta_end - beta_start) * progress
            else:
                return beta_end - (beta_end - beta_start) * progress
        
        return beta_end
    
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        beta = self.get_beta(epoch)
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_images, mu, logvar = self.model(images)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta=beta)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Accumulate stats
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Recon', recon_loss.item(), self.global_step)
                self.writer.add_scalar('Train/KL', kl_loss.item(), self.global_step)
                self.writer.add_scalar('Train/Beta', beta, self.global_step)
            
            self.global_step += 1
            pbar.update(1)
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'recon': total_recon / (batch_idx + 1),
                'kl': total_kl / (batch_idx + 1),
                'beta': f'{beta:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        
        return avg_loss, avg_recon, avg_kl
    
    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> Tuple[float, float, float]:
        """Validate on validation set."""
        self.model.eval()
        beta = self.get_beta(epoch)
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for images, _ in pbar:
            images = images.to(self.device)
            
            # Forward pass
            recon_images, mu, logvar = self.model(images)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta=beta)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            pbar.update(1)
        
        avg_loss = total_loss / len(val_loader)
        avg_recon = total_recon / len(val_loader)
        avg_kl = total_kl / len(val_loader)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.writer.add_scalar('Val/Recon', avg_recon, epoch)
            self.writer.add_scalar('Val/KL', avg_kl, epoch)
        
        return avg_loss, avg_recon, avg_kl
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config.__dict__,
        }
        
        # Save latest checkpoint
        ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self, train_loader, val_loader):
        """Full training loop."""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.training.num_epochs):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader, epoch)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kl'].append(train_kl)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kl'].append(val_kl)
            self.history['beta'].append(self.get_beta(epoch))
            
            # Log epoch
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.num_epochs} | "
                f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
                f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})"
            )
            
            # Save checkpoints
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, is_best=True)
        
        # Save final training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training completed! History saved to {history_path}")


def main(config: Optional[Config] = None):
    """Main training script."""
    if config is None:
        config = DEFAULT_CONFIG
    
    logger.info(config)
    
    # Create model
    model = ConvVAE(
        latent_dim=config.model.latent_dim,
        image_channels=config.model.image_channels
    )
    
    # Create dataloaders (using mock data for demonstration)
    train_loader, val_loader, test_loader = get_dataloaders(config, mock=True)
    
    # Create trainer and train
    trainer = Trainer(config, model)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
