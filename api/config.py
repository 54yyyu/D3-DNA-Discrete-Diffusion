"""
Configuration management for D3-DNA API
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


class APIConfig:
    """Configuration for the D3-DNA API"""
    
    def __init__(self):
        # Project root (parent of api directory)
        self.project_root = Path(__file__).parent.parent
        
        # Server configuration
        self.host = os.getenv("API_HOST", "localhost")
        self.port = int(os.getenv("API_PORT", "2025"))
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_memory_fraction = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
        
        # Model paths and configurations
        self.model_configs = {
            "deepstarr": {
                # SET THESE TO YOUR ACTUAL FILES
                "config_file": "model_zoo/deepstarr/configs/transformer.yaml",  # YOUR CONFIG FILE
                "checkpoint": "experiments/deepstarr/checkpoints/model-epoch=279-val_loss=319.2632.ckpt",  # YOUR MODEL CHECKPOINT
                "oracle_checkpoint": "model_zoo/deepstarr/oracle_models/oracle_DeepSTARR_DeepSTARR_data.ckpt",  # YOUR ORACLE MODEL CHECKPOINT
                "data_file": "model_zoo/deepstarr/DeepSTARR_data.h5",  # YOUR DATA FILE
                
                "sequence_length": 249,
                "signal_dim": 2,
                "num_classes": 4,
                "description": "DeepSTARR enhancer activity prediction dataset"
            }
        }
        
        # Load dataset configurations
        self._load_dataset_configs()
        
        # API configuration
        self.max_samples_per_request = int(os.getenv("MAX_SAMPLES_PER_REQUEST", "1000"))
        self.max_steps_per_request = int(os.getenv("MAX_STEPS_PER_REQUEST", "500"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
        
        # CORS configuration
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        
    def _load_dataset_configs(self):
        """Load dataset-specific configurations from YAML files"""
        for dataset_name, config in self.model_configs.items():
            try:
                # Convert relative paths to absolute paths
                config["config_file"] = self.project_root / config["config_file"]
                config["checkpoint"] = self.project_root / config["checkpoint"]
                config["oracle_checkpoint"] = self.project_root / config["oracle_checkpoint"]
                config["data_file"] = self.project_root / config["data_file"]
                
                # Load the config and extract architecture
                if config["config_file"].exists():
                    loaded_config = OmegaConf.load(config["config_file"])
                    config["config"] = loaded_config
                    # Read architecture from config file
                    config["architecture"] = loaded_config.model.architecture
                    print(f"Loaded {dataset_name} config: {config['architecture']} architecture")
                else:
                    raise ValueError(f"Config file not found: {config['config_file']}")
                
            except Exception as e:
                print(f"Warning: Failed to process config for {dataset_name}: {e}")
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset"""
        if dataset_name not in self.model_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        config = self.model_configs[dataset_name].copy()
        
        # Architecture is already loaded from config file
        # Checkpoint path is set during initialization
        config["checkpoint_path"] = config["checkpoint"]
            
        return config
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available datasets"""
        datasets_info = {}
        for name, config in self.model_configs.items():
            datasets_info[name] = {
                "sequence_length": config["sequence_length"],
                "signal_dim": config["signal_dim"],
                "num_classes": config["num_classes"],
                "description": config["description"],
                "architecture": config.get("architecture", "transformer"),  # Architecture from config file
                "default_steps": config["sequence_length"]
            }
        return datasets_info
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all required paths exist"""
        validation_results = {}
        
        for dataset_name, config in self.model_configs.items():
            dataset_valid = True
            
            # Check config file
            if not config["config_file"].exists():
                print(f"Warning: Config file missing for {dataset_name}: {config['config_file']}")
                dataset_valid = False
                
            # Check data file
            if config.get("data_file") and not config["data_file"].exists():
                print(f"Warning: Data file missing for {dataset_name}: {config['data_file']}")
                dataset_valid = False
                
            # Check model checkpoint
            if config.get("checkpoint") and not config["checkpoint"].exists():
                print(f"Warning: Model checkpoint missing for {dataset_name}: {config['checkpoint']}")
                dataset_valid = False
                
            # Check oracle checkpoint
            if config.get("oracle_checkpoint") and not config["oracle_checkpoint"].exists():
                print(f"Warning: Oracle checkpoint missing for {dataset_name}: {config['oracle_checkpoint']}")
                dataset_valid = False
                
            validation_results[dataset_name] = dataset_valid
            
        return validation_results


# Global configuration instance
config = APIConfig()