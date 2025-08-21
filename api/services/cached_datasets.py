"""
Cached dataset loading utilities for D3-DNA API
"""

import sys
from pathlib import Path
from typing import Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_zoo.deepstarr.data import DeepSTARRDataset
from .model_loader import model_loader


def get_deepstarr_datasets_cached(h5_file_path: str) -> Tuple:
    """
    Get DeepSTARR datasets with caching to avoid reloading from disk.
    Only loads test set since that's all we need for API generation.
    
    Args:
        h5_file_path: Path to the HDF5 data file
        
    Returns:
        Tuple of (None, None, test_dataset) - only test set is loaded
    """
    
    # Try to get from cache first
    cached_datasets = model_loader.get_cached_dataset_objects(h5_file_path)
    if cached_datasets is not None:
        print(f"✓ Using cached dataset objects for: {h5_file_path}")
        return cached_datasets
    
    # Only load test set - we don't need train/valid for API generation
    print(f"Loading test dataset from: {h5_file_path}")
    test_set = DeepSTARRDataset(h5_file_path, split='test')
    
    # Cache with None for train/valid sets
    datasets = (None, None, test_set)
    model_loader.cache_dataset_objects(h5_file_path, *datasets)
    
    return datasets