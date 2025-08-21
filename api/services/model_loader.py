"""
Model loading service with GPU support for D3-DNA API
"""

import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from omegaconf import OmegaConf

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_zoo.deepstarr.models import load_trained_model
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR, load_evoaug_oracle_model


class ModelLoader:
    """Handles loading and caching of D3-DNA models with GPU support"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.loading_times: Dict[str, float] = {}
        
        print(f"ModelLoader initialized on device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_model_key(self, dataset: str, architecture: str) -> str:
        """Generate a unique key for model caching"""
        return f"{dataset}_{architecture}"
    
    def is_model_loaded(self, dataset: str, architecture: str) -> bool:
        """Check if a model is already loaded"""
        key = self.get_model_key(dataset, architecture)
        return key in self.loaded_models
    
    def load_deepstarr_model(self, config: OmegaConf, architecture: str, 
                           checkpoint_path: str) -> Tuple[Any, Any, Any]:
        """Load DeepSTARR diffusion model"""
        
        print(f"Loading DeepSTARR {architecture} model from {checkpoint_path}")
        
        # Load model using existing model loading logic
        model, graph, noise = load_trained_model(str(checkpoint_path), config, architecture, self.device)
        
        return model, graph, noise
    
    def load_deepstarr_oracle(self, oracle_checkpoint: str, data_path: str, 
                            use_evoaug: bool = False) -> Any:
        """Load DeepSTARR oracle model"""
        print(f"Loading DeepSTARR oracle model from {oracle_checkpoint}")
        
        try:
            if use_evoaug:
                oracle = load_evoaug_oracle_model(oracle_checkpoint, device=self.device)
                print("✓ Loaded EvoAug DeepSTARR oracle model")
            else:
                oracle = PL_DeepSTARR.load_from_checkpoint(
                    oracle_checkpoint, 
                    input_h5_file=str(data_path)
                ).eval()
                oracle.to(self.device)
                print("✓ Loaded standard DeepSTARR oracle model")
                
            return oracle
        except Exception as e:
            print(f"Failed to load DeepSTARR oracle model: {e}")
            return None
    
    def load_model(self, dataset: str, dataset_config: Dict[str, Any],
                  force_reload: bool = False) -> Dict[str, Any]:
        """Load a model and cache it"""
        
        # Get architecture from dataset config
        architecture = dataset_config["architecture"]
        key = self.get_model_key(dataset, architecture)
        
        # Return cached model if available and not forcing reload
        if not force_reload and key in self.loaded_models:
            print(f"Using cached model: {key}")
            return self.loaded_models[key]
        
        start_time = time.time()
        
        try:
            if dataset.lower() == "deepstarr":
                # Get checkpoint path from dataset config
                checkpoint_path = dataset_config["checkpoint_path"]
                
                # Load diffusion model
                model, graph, noise = self.load_deepstarr_model(
                    dataset_config["config"], 
                    architecture, 
                    checkpoint_path
                )
                
                # Load oracle model
                oracle = None
                if dataset_config.get("oracle_checkpoint"):
                    use_evoaug = dataset_config["config"].get("eval", {}).get("use_evoaug_oracle", False)
                    oracle = self.load_deepstarr_oracle(
                        str(dataset_config["oracle_checkpoint"]),
                        str(dataset_config["data_file"]),
                        use_evoaug
                    )
                
                # Cache the loaded models
                model_data = {
                    "model": model,
                    "graph": graph, 
                    "noise": noise,
                    "oracle": oracle,
                    "config": dataset_config["config"],
                    "dataset": dataset,
                    "architecture": architecture,
                    "device": self.device,
                    "loaded_at": time.time()
                }
                
                self.loaded_models[key] = model_data
                loading_time = time.time() - start_time
                self.loading_times[key] = loading_time
                
                print(f"✓ Model loaded successfully: {key} (took {loading_time:.2f}s)")
                return model_data
                
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
                
        except Exception as e:
            error_msg = f"Failed to load model {key}: {str(e)}"
            print(error_msg)
            
            # Store error info
            self.loaded_models[key] = {
                "error": error_msg,
                "dataset": dataset,
                "architecture": architecture,
                "loaded_at": time.time()
            }
            
            raise Exception(error_msg)
    
    def get_model(self, dataset: str, architecture: str) -> Dict[str, Any]:
        """Get a loaded model"""
        key = self.get_model_key(dataset, architecture)
        
        if key not in self.loaded_models:
            raise ValueError(f"Model not loaded: {key}")
            
        model_data = self.loaded_models[key]
        
        if "error" in model_data:
            raise Exception(f"Model failed to load: {model_data['error']}")
            
        return model_data
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "models": []
        }
        
        if torch.cuda.is_available():
            status["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(0) / 1e9,
                "cached": torch.cuda.memory_reserved(0) / 1e9,
                "total": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        
        for key, model_data in self.loaded_models.items():
            model_status = {
                "key": key,
                "dataset": model_data.get("dataset"),
                "architecture": model_data.get("architecture"),
                "loaded": "error" not in model_data,
                "loading_time": self.loading_times.get(key),
                "error": model_data.get("error")
            }
            
            if "oracle" in model_data:
                model_status["oracle_loaded"] = model_data["oracle"] is not None
            
            status["models"].append(model_status)
        
        return status
    
    def cleanup(self):
        """Clean up loaded models to free memory"""
        for key in self.loaded_models:
            if "model" in self.loaded_models[key]:
                del self.loaded_models[key]["model"]
                
        self.loaded_models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Model cache cleared")


# Global model loader instance
model_loader = ModelLoader()