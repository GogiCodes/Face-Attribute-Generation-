"""
CelebA-VAE-Latent-Manipulator Package

A deep generative model for facial attribute generation and manipulation
through convolutional variational autoencoders and latent space arithmetic.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Conv-VAE for Facial Attribute Generation"

from .model import ConvVAE, Encoder, Decoder, vae_loss
from .config import Config, DEFAULT_CONFIG
from .data_loader import CelebADataset, MockCelebADataset, get_dataloaders
from .train import Trainer
from .evaluate import ModelEvaluator
from .manipulate import LatentManipulator

__all__ = [
    'ConvVAE',
    'Encoder',
    'Decoder',
    'vae_loss',
    'Config',
    'DEFAULT_CONFIG',
    'CelebADataset',
    'MockCelebADataset',
    'get_dataloaders',
    'Trainer',
    'ModelEvaluator',
    'LatentManipulator',
]
