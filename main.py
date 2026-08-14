"""
Main entry point for Conv-VAE project.
Provides CLI interface for training, evaluation, and manipulation.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from config import Config, DEFAULT_CONFIG
from model import ConvVAE
from data_loader import get_dataloaders
from train import Trainer
from evaluate import ModelEvaluator
from manipulate import LatentManipulator, demonstrate_manipulation


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConvVAEPipeline:
    """Complete Conv-VAE pipeline manager."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = ConvVAE(
            latent_dim=self.config.model.latent_dim,
            image_channels=self.config.model.image_channels
        )
    
    def train(self, use_mock_data: bool = True):
        """Train the model."""
        logger.info("Starting training pipeline...")
        logger.info(self.config)
        
        # Load data
        train_loader, val_loader, test_loader = get_dataloaders(
            self.config, 
            mock=use_mock_data
        )
        
        # Create trainer
        trainer = Trainer(self.config, self.model)
        
        # Train
        trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        evaluator = ModelEvaluator(self.model, self.config)
        results = evaluator.full_evaluation(test_loader)
        evaluator.print_evaluation_report(results)
        
        logger.info("Training completed!")
        return trainer, results
    
    def evaluate(self, checkpoint_path: str):
        """Evaluate a checkpoint."""
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Load test data
        _, _, test_loader = get_dataloaders(self.config, mock=True)
        
        # Evaluate
        evaluator = ModelEvaluator(self.model, self.config)
        results = evaluator.full_evaluation(test_loader)
        evaluator.print_evaluation_report(results)
        
        return results
    
    def demonstrate_generation(self):
        """Demonstrate random face generation."""
        logger.info("Generating random faces...")
        evaluator = ModelEvaluator(self.model, self.config)
        
        samples = evaluator.generate_samples(num_samples=8)
        logger.info(f"Generated {samples.shape[0]} samples of shape {samples.shape[1:]}")
        
        return samples
    
    def demonstrate_manipulation(self):
        """Demonstrate latent space manipulation."""
        logger.info("Demonstrating latent space manipulation...")
        manipulator = LatentManipulator(self.model, self.config)
        
        # Example: generate a random face and manipulate it
        z = torch.randn(1, self.config.model.latent_dim)
        face = manipulator.decode_latent(z)
        
        logger.info(f"Generated random face of shape {face.shape}")
        logger.info("\nAvailable attributes for manipulation:")
        for attr in self.config.manipulation.attribute_list:
            logger.info(f"  - {attr}")
        
        return face


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CelebA-VAE-Latent-Manipulator: Facial Attribute Generation"
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'generate', 'manipulate', 'demo'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cpu or cuda)'
    )
    
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real CelebA dataset instead of mock data'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = DEFAULT_CONFIG
    
    config.device = args.device
    
    # Create pipeline
    pipeline = ConvVAEPipeline(config)
    
    # Execute command
    if args.command == 'train':
        logger.info("🚀 Starting training...")
        pipeline.train(use_mock_data=not args.use_real_data)
    
    elif args.command == 'evaluate':
        if not args.checkpoint:
            logger.error("Checkpoint path required for evaluation")
            return
        pipeline.evaluate(args.checkpoint)
    
    elif args.command == 'generate':
        logger.info("🎨 Generating random faces...")
        samples = pipeline.demonstrate_generation()
        logger.info(f"✓ Generated {samples.shape[0]} samples")
    
    elif args.command == 'manipulate':
        logger.info("✨ Demonstrating latent space manipulation...")
        face = pipeline.demonstrate_manipulation()
        logger.info("✓ Manipulation ready")
    
    elif args.command == 'demo':
        logger.info("🎭 Running full demonstration...")
        logger.info("\n1. Generating random faces...")
        samples = pipeline.demonstrate_generation()
        
        logger.info("\n2. Demonstrating manipulation capabilities...")
        pipeline.demonstrate_manipulation()
        
        logger.info("\n✓ Demonstration complete!")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
