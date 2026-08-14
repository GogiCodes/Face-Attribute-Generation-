"""
Latent space manipulation for facial attribute editing using vector arithmetic.
Supports extracting attribute directions, interpolation, and controlled editing.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from PIL import Image
import json

from model import ConvVAE
from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class LatentManipulator:
    """Perform semantic edits in latent space using attribute vectors."""
    
    def __init__(self, model: ConvVAE, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.attribute_vectors: Dict[str, torch.Tensor] = {}
        self.attribute_statistics = {}
    
    def extract_attribute_vectors(self, 
                                 latent_codes: torch.Tensor,
                                 attributes: torch.Tensor,
                                 attribute_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract attribute direction vectors using difference between samples.
        
        Vector for attribute i: mean(z | attr_i=1) - mean(z | attr_i=0)
        
        Args:
            latent_codes: Latent vectors (N, latent_dim)
            attributes: Attribute matrix (N, num_attributes)
            attribute_list: Names of attributes
        
        Returns:
            Dictionary mapping attribute name to direction vector
        """
        attribute_vectors = {}
        
        for idx, attr_name in enumerate(attribute_list):
            # Get samples with and without this attribute
            has_attr = attributes[:, idx] > 0.5
            no_attr = ~has_attr
            
            if has_attr.sum() > 0 and no_attr.sum() > 0:
                # Compute mean latent vector for each group
                z_with = latent_codes[has_attr].mean(dim=0)
                z_without = latent_codes[no_attr].mean(dim=0)
                
                # Attribute vector is the difference
                v_attr = z_with - z_without
                
                # Store normalized vector
                v_attr_norm = v_attr / (v_attr.norm() + 1e-8)
                attribute_vectors[attr_name] = v_attr_norm
                
                logger.info(f"Extracted vector for '{attr_name}': "
                          f"norm={v_attr.norm():.4f}, "
                          f"samples: {has_attr.sum()} with, {no_attr.sum()} without")
        
        self.attribute_vectors = attribute_vectors
        return attribute_vectors
    
    def manipulate_latent(self, z: torch.Tensor, 
                         attribute: str, 
                         strength: float) -> torch.Tensor:
        """
        Manipulate latent vector by adding scaled attribute vector.
        
        z' = z + strength * v_attribute
        
        Args:
            z: Original latent vector (latent_dim,)
            attribute: Name of attribute to modify
            strength: Strength of manipulation (typically -3.0 to 3.0)
        
        Returns:
            Manipulated latent vector
        """
        if attribute not in self.attribute_vectors:
            raise ValueError(f"Attribute '{attribute}' not found. "
                           f"Available: {list(self.attribute_vectors.keys())}")
        
        v_attr = self.attribute_vectors[attribute]
        z_manipulated = z + strength * v_attr
        
        return z_manipulated
    
    def interpolate_latent(self, z1: torch.Tensor, z2: torch.Tensor, 
                          steps: int = 10) -> torch.Tensor:
        """
        Linear interpolation in latent space between two points.
        
        z(t) = (1-t)*z1 + t*z2, where t in [0, 1]
        
        Args:
            z1: Start latent vector
            z2: End latent vector
            steps: Number of interpolation steps
        
        Returns:
            Interpolated latent vectors (steps, latent_dim)
        """
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        z_interp = []
        
        for alpha in alphas:
            z_t = (1 - alpha) * z1 + alpha * z2
            z_interp.append(z_t)
        
        return torch.stack(z_interp)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent vector.
        
        Args:
            image: Image tensor (3, H, W) or (B, 3, H, W)
        
        Returns:
            Latent vector (latent_dim,) or (B, latent_dim)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        with torch.no_grad():
            z = self.model.encode(image)
        
        return z.squeeze(0) if z.shape[0] == 1 else z
    
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector (latent_dim,) or (B, latent_dim)
        
        Returns:
            Reconstructed image (3, H, W) or (B, 3, H, W)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        z = z.to(self.device)
        with torch.no_grad():
            image = self.model.decode(z)
        
        return image.squeeze(0) if image.shape[0] == 1 else image
    
    def edit_face(self, image: torch.Tensor, 
                 edits: Dict[str, float]) -> torch.Tensor:
        """
        Apply multiple attribute edits to an image.
        
        Args:
            image: Input image tensor (3, H, W)
            edits: Dictionary mapping attribute name to strength value
        
        Returns:
            Edited image tensor (3, H, W)
        """
        # Encode
        z = self.encode_image(image)
        
        # Apply edits
        for attribute, strength in edits.items():
            z = self.manipulate_latent(z, attribute, strength)
        
        # Decode
        edited_image = self.decode_latent(z)
        
        return edited_image
    
    def generate_attribute_showcase(self, image: torch.Tensor,
                                   attribute: str,
                                   num_steps: int = 7) -> torch.Tensor:
        """
        Generate a grid showing attribute interpolation.
        
        Args:
            image: Input image
            attribute: Attribute to interpolate
            num_steps: Number of steps from -3 to +3
        
        Returns:
            Grid of edited images (num_steps, 3, H, W)
        """
        z = self.encode_image(image)
        
        alpha_values = np.linspace(-3.0, 3.0, num_steps)
        edited_images = []
        
        for alpha in alpha_values:
            z_edit = self.manipulate_latent(z, attribute, alpha)
            img_edit = self.decode_latent(z_edit)
            edited_images.append(img_edit)
        
        return torch.stack(edited_images)
    
    def face_morph(self, image1: torch.Tensor, image2: torch.Tensor,
                   steps: int = 10) -> torch.Tensor:
        """
        Smooth morphing between two faces in latent space.
        
        Args:
            image1: First face image
            image2: Second face image
            steps: Number of interpolation steps
        
        Returns:
            Morphing sequence (steps, 3, H, W)
        """
        z1 = self.encode_image(image1)
        z2 = self.encode_image(image2)
        
        # Remove batch dimension if present
        if z1.dim() > 1:
            z1 = z1[0]
        if z2.dim() > 1:
            z2 = z2[0]
        
        # Interpolate
        z_interp = self.interpolate_latent(z1, z2, steps)
        
        # Decode all
        images = []
        for z in z_interp:
            img = self.decode_latent(z)
            if img.dim() == 4:
                img = img[0]
            images.append(img)
        
        return torch.stack(images)
    
    def save_statistics(self, output_path: str):
        """Save attribute vector statistics to file."""
        stats = {}
        for attr, v in self.attribute_vectors.items():
            stats[attr] = {
                'norm': float(v.norm().item()),
                'mean': float(v.mean().item()),
                'std': float(v.std().item()),
            }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved attribute statistics to {output_path}")


def demonstrate_manipulation(checkpoint_path: Optional[str] = None,
                             config: Optional[Config] = None):
    """
    Demonstrate latent space manipulation capabilities.
    
    This is a placeholder that shows how to use the LatentManipulator class.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    logger.info("Initializing Conv-VAE for manipulation...")
    
    # Create model
    model = ConvVAE(
        latent_dim=config.model.latent_dim,
        image_channels=config.model.image_channels
    )
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.info("Using untrained model for demonstration")
    
    # Create manipulator
    manipulator = LatentManipulator(model, config)
    
    logger.info("Latent manipulator ready!")
    logger.info("\nAvailable attributes:")
    for i, attr in enumerate(config.manipulation.attribute_list):
        logger.info(f"  {i+1}. {attr}")
    
    # Generate random latent vector and decode
    z_random = torch.randn(1, config.model.latent_dim)
    random_face = manipulator.decode_latent(z_random)
    
    logger.info("\nGenerated random face from prior N(0, I)")
    logger.info(f"Image shape: {random_face.squeeze(0).shape}")
    
    # Example: if attribute vectors were extracted
    # we could demonstrate face editing and morphing


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demonstrate_manipulation()
