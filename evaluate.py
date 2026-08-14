"""
Evaluation and metrics for Conv-VAE model performance.
Computes reconstruction quality, interpolation smoothness, and attribute consistency.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from torch.utils.data import DataLoader

from model import ConvVAE, vae_loss
from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance metrics."""
    
    def __init__(self, model: ConvVAE, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_reconstruction(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate reconstruction quality on test set.
        
        Metrics:
        - MSE (Mean Squared Error)
        - KL Divergence
        - Total VAE Loss
        """
        total_mse = 0.0
        total_kl = 0.0
        total_vae = 0.0
        num_batches = 0
        
        for images, _ in test_loader:
            images = images.to(self.device)
            
            # Forward pass
            recon_images, mu, logvar = self.model(images)
            
            # Compute losses
            vae_total, mse, kl = vae_loss(recon_images, images, mu, logvar, beta=1.0)
            
            total_mse += mse.item()
            total_kl += kl.item()
            total_vae += vae_total.item()
            num_batches += 1
        
        metrics = {
            'mse': total_mse / num_batches,
            'kl_divergence': total_kl / num_batches,
            'vae_loss': total_vae / num_batches,
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate_interpolation_smoothness(self, test_loader: DataLoader,
                                         num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate smoothness of latent space interpolation.
        
        Metrics:
        - Reconstruction consistency ratio (how similar interpolated frames are)
        """
        sample_count = 0
        total_interpolation_error = 0.0
        
        for images, _ in test_loader:
            if sample_count >= num_samples:
                break
            
            images = images.to(self.device)
            batch_size = min(images.shape[0], num_samples - sample_count)
            images = images[:batch_size]
            
            # Encode
            mu, _ = self.model.encoder(images)
            
            # Create pairs and interpolate
            for i in range(batch_size - 1):
                z1 = mu[i]
                z2 = mu[i + 1]
                
                # Linear interpolation
                alphas = torch.linspace(0, 1, 5, device=self.device)
                recon_errors = []
                
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img_interp = self.model.decode(z_interp.unsqueeze(0))
                    
                    # Reconstruction error at interpolated point
                    error = torch.mean((img_interp - images[i:i+1]) ** 2).item()
                    recon_errors.append(error)
                
                # Smoothness: variance of reconstruction errors should be low
                interpolation_error = np.var(recon_errors)
                total_interpolation_error += interpolation_error
            
            sample_count += batch_size
        
        avg_smoothness = total_interpolation_error / max(sample_count - 1, 1)
        
        return {
            'interpolation_variance': avg_smoothness,
        }
    
    @torch.no_grad()
    def evaluate_latent_distribution(self, test_loader: DataLoader,
                                    num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate latent space distribution properties.
        
        Metrics:
        - Mean and std of latent codes
        - Coverage (how well the latent space is utilized)
        """
        latent_codes = []
        sample_count = 0
        
        for images, _ in test_loader:
            if sample_count >= num_samples:
                break
            
            images = images.to(self.device)
            mu, _ = self.model.encoder(images)
            latent_codes.append(mu.cpu())
            sample_count += images.shape[0]
        
        latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
        
        metrics = {
            'latent_mean': float(latent_codes.mean().item()),
            'latent_std': float(latent_codes.std().item()),
            'latent_min': float(latent_codes.min().item()),
            'latent_max': float(latent_codes.max().item()),
        }
        
        return metrics
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 8) -> torch.Tensor:
        """
        Generate random samples from the model.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Generated images (num_samples, 3, H, W)
        """
        z = torch.randn(num_samples, self.config.model.latent_dim, device=self.device)
        samples = self.model.decode(z)
        
        return samples.cpu()
    
    def full_evaluation(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Run full evaluation suite.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        results = {}
        
        # Reconstruction quality
        logger.info("Evaluating reconstruction quality...")
        results['reconstruction'] = self.evaluate_reconstruction(test_loader)
        
        # Interpolation smoothness
        logger.info("Evaluating interpolation smoothness...")
        results['interpolation'] = self.evaluate_interpolation_smoothness(test_loader)
        
        # Latent distribution
        logger.info("Evaluating latent distribution...")
        results['latent_space'] = self.evaluate_latent_distribution(test_loader)
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Dict[str, float]]):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("CONV-VAE EVALUATION REPORT")
        print("=" * 60)
        
        for category, metrics in results.items():
            print(f"\n{category.upper()}:")
            print("-" * 40)
            for metric_name, value in metrics.items():
                print(f"  {metric_name:.<30} {value:.6f}")
        
        print("\n" + "=" * 60)


def evaluate_checkpoint(checkpoint_path: str, 
                       test_loader: DataLoader,
                       config: Optional[Config] = None):
    """
    Evaluate a trained model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        test_loader: Test dataset loader
        config: Model configuration
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Load model
    model = ConvVAE(
        latent_dim=config.model.latent_dim,
        image_channels=config.model.image_channels
    )
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # Evaluate
    evaluator = ModelEvaluator(model, config)
    results = evaluator.full_evaluation(test_loader)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluation module ready. Use evaluate_checkpoint() to evaluate a model.")
