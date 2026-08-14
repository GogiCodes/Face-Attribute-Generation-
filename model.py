"""
Convolutional Variational Autoencoder (Conv-VAE) for facial attribute generation.
Architecture optimized for CelebA dataset with improved reconstruction fidelity.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """Convolutional encoder that maps images to latent space."""
    
    def __init__(self, latent_dim: int = 64, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: progressively downsample spatial dimensions
        self.enc1 = ConvBlock(image_channels, 32, kernel_size=4, stride=2)      # 64x64 -> 32x32
        self.enc2 = ConvBlock(32, 64, kernel_size=4, stride=2)                  # 32x32 -> 16x16
        self.enc3 = ConvBlock(64, 128, kernel_size=4, stride=2)                 # 16x16 -> 8x8
        self.enc4 = ConvBlock(128, 256, kernel_size=4, stride=2)                # 8x8 -> 4x4
        
        # Flatten and project to latent distribution parameters
        self.fc_input_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: Input images (batch_size, 3, 64, 64)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """Convolutional decoder that maps latent vectors back to image space."""
    
    def __init__(self, latent_dim: int = 64, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project latent vector to spatial dimensions
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder: progressively upsample spatial dimensions
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.dec4 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vectors (batch_size, latent_dim)
        
        Returns:
            Reconstructed images (batch_size, 3, 64, 64)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        
        x = torch.relu(self.bn1(self.dec1(x)))
        x = torch.relu(self.bn2(self.dec2(x)))
        x = torch.relu(self.bn3(self.dec3(x)))
        x = torch.sigmoid(self.dec4(x))  # Output in [0, 1]
        
        return x


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder combining encoder and decoder.
    Implements reparameterization trick for differentiable sampling.
    """
    
    def __init__(self, latent_dim: int = 64, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, image_channels)
        self.decoder = Decoder(latent_dim, image_channels)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, exp(logvar)) using auxiliary noise.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input images
        
        Returns:
            recon_x: Reconstructed images
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent vector (deterministic)."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample random images from prior N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
             logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute β-VAE loss combining reconstruction (MSE) and KL divergence.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term (β-VAE)
    
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE)
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with β weighting
    total_loss = mse_loss + beta * kl_loss
    
    return total_loss, mse_loss, kl_loss
