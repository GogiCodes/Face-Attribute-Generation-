"""
Configuration settings for Conv-VAE training and inference.
Optimized for CelebA dataset with 64x64 resolution.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = "data/celeba"
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    num_samples_for_stats: int = 1000  # For computing attribute statistics


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    latent_dim: int = 64
    image_channels: int = 3
    encoder_layers: int = 4
    decoder_layers: int = 4
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 50
    learning_rate: float = 1e-3
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_schedule: str = "linear"  # Options: "constant", "linear", "cyclical"
    beta_warmup_epochs: int = 10
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    checkpoint_interval: int = 5
    enable_tensorboard: bool = True


@dataclass
class ManipulationConfig:
    """Latent space manipulation configuration."""
    attribute_list: list = field(default_factory=lambda: [
        "Male", "Smiling", "Eyeglasses", "Wearing_Hat", 
        "Young", "Attractive", "Pale_Skin", "Heavy_Makeup"
    ])
    num_samples_per_attribute: int = 500
    interpolation_steps: int = 10
    alpha_range: tuple = (-3.0, 3.0)  # Range for attribute strength manipulation


@dataclass
class Config:
    """Master configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    manipulation: ManipulationConfig = field(default_factory=ManipulationConfig)
    
    # General settings
    seed: int = 42
    device: str = "cuda"
    project_name: str = "celeba-vae-latent-manipulator"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'manipulation': self.manipulation.__dict__,
            'seed': self.seed,
            'device': self.device,
            'project_name': self.project_name,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = [
            "=" * 60,
            "Conv-VAE Configuration",
            "=" * 60,
            f"Data Config:\n  {self.data}",
            f"Model Config:\n  {self.model}",
            f"Training Config:\n  {self.training}",
            f"Manipulation Config:\n  {self.manipulation}",
            f"Seed: {self.seed}",
            f"Device: {self.device}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Default configuration instance
DEFAULT_CONFIG = Config()
