"""
Data loading utilities for CelebA dataset with attribute support.
Handles image preprocessing, splits, and attribute vector extraction.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Tuple, List, Optional
import os


class CelebADataset(Dataset):
    """
    CelebA dataset with facial attributes.
    Expects directory structure: data/celeba/img_align_celeba/ and list_attr_celeba.txt
    """
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 image_size: int = 64,
                 attribute_list: Optional[List[str]] = None,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Path to CelebA dataset root
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to
            attribute_list: List of attributes to extract
            transform: Torchvision transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.img_dir = self.root_dir / "img_align_celeba"
        self.attr_file = self.root_dir / "list_attr_celeba.txt"
        
        # Load attributes
        self.attributes = self._load_attributes(attribute_list)
        
        # Create split indices
        self.indices = self._get_split_indices()
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def _load_attributes(self, attribute_list: Optional[List[str]]) -> pd.DataFrame:
        """Load CelebA attribute file."""
        if not self.attr_file.exists():
            raise FileNotFoundError(f"Attribute file not found: {self.attr_file}")
        
        # Read attributes (skip first 2 rows which contain metadata)
        df = pd.read_csv(self.attr_file, sep="\s+", skiprows=2)
        
        # Convert -1 and 1 to 0 and 1
        df = (df + 1) // 2
        
        if attribute_list:
            df = df[attribute_list]
        
        return df
    
    def _get_split_indices(self) -> np.ndarray:
        """Get indices for train/val/test split (80/10/10)."""
        total_samples = len(self.attributes)
        indices = np.arange(total_samples)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        if self.split == 'train':
            return indices[:int(0.8 * total_samples)]
        elif self.split == 'val':
            return indices[int(0.8 * total_samples):int(0.9 * total_samples)]
        else:  # test
            return indices[int(0.9 * total_samples):]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and attributes.
        
        Returns:
            image: Tensor of shape (3, H, W)
            attributes: Tensor of shape (num_attributes,)
        """
        actual_idx = self.indices[idx]
        img_name = self.attributes.index[actual_idx]
        img_path = self.img_dir / img_name
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get attributes for this image
        attributes = torch.tensor(
            self.attributes.iloc[actual_idx].values, 
            dtype=torch.float32
        )
        
        return image, attributes


class MockCelebADataset(Dataset):
    """Mock CelebA dataset for testing without downloading actual data."""
    
    def __init__(self, num_samples: int = 1000, 
                 image_size: int = 64, 
                 num_attributes: int = 8):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_attributes = num_attributes
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random image and attributes."""
        # Random RGB image
        image = torch.rand(3, self.image_size, self.image_size)
        image = self.transform(image)
        
        # Random binary attributes
        attributes = torch.randint(0, 2, (self.num_attributes,), dtype=torch.float32)
        
        return image, attributes


def get_dataloaders(config,
                    mock: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        config: Configuration object
        mock: If True, use mock dataset for testing
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if mock:
        train_ds = MockCelebADataset(
            num_samples=int(config.data.batch_size * 20),
            image_size=config.data.image_size,
            num_attributes=len(config.manipulation.attribute_list)
        )
        val_ds = MockCelebADataset(
            num_samples=int(config.data.batch_size * 5),
            image_size=config.data.image_size,
            num_attributes=len(config.manipulation.attribute_list)
        )
        test_ds = MockCelebADataset(
            num_samples=int(config.data.batch_size * 5),
            image_size=config.data.image_size,
            num_attributes=len(config.manipulation.attribute_list)
        )
    else:
        train_ds = CelebADataset(
            root_dir=config.data.dataset_path,
            split='train',
            image_size=config.data.image_size,
            attribute_list=config.manipulation.attribute_list
        )
        val_ds = CelebADataset(
            root_dir=config.data.dataset_path,
            split='val',
            image_size=config.data.image_size,
            attribute_list=config.manipulation.attribute_list
        )
        test_ds = CelebADataset(
            root_dir=config.data.dataset_path,
            split='test',
            image_size=config.data.image_size,
            attribute_list=config.manipulation.attribute_list
        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    return train_loader, val_loader, test_loader
