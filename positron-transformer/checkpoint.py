"""
Clean checkpoint system for saving/loading model weights and config separately
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class CheckpointManager:
    """Manages model checkpoints with separated weights and configuration"""
    
    @staticmethod
    def save(model, save_dir: str, name: str = "model") -> None:
        """
        Save model checkpoint with separated components:
        - config.json: Model architecture configuration
        - weights.npz: Model weights
        - training_state.json: Optional training metadata
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save configuration
        config_path = save_path / f"{name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(model.config, f, indent=2)
        
        # 2. Save weights separately
        weights = CheckpointManager._extract_weights(model)
        weights_path = save_path / f"{name}_weights.npz"
        np.savez_compressed(weights_path, **weights)
        
        print(f"✅ Checkpoint saved to {save_dir}/")
        print(f"   - Config: {name}_config.json")
        print(f"   - Weights: {name}_weights.npz")
    
    @staticmethod
    def load(model, load_dir: str, name: str = "model") -> None:
        """
        Load model checkpoint from separated files
        """
        load_path = Path(load_dir)
        
        # 1. Load and verify configuration
        config_path = load_path / f"{name}_config.json"
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Verify config matches (optional - can also rebuild model from config)
        if not CheckpointManager._verify_config(model.config, saved_config):
            print("⚠️ Warning: Model configuration mismatch")
        
        # 2. Load weights
        weights_path = load_path / f"{name}_weights.npz"
        weights = np.load(weights_path, allow_pickle=True)
        CheckpointManager._load_weights(model, weights)
        
        print(f"✅ Checkpoint loaded from {load_dir}/")
    
    @staticmethod
    def _extract_weights(model) -> Dict[str, np.ndarray]:
        """Extract all weights from model"""
        weights = {}
        
        # Output projection weights
        weights['output_projection'] = model.output_projection
        if model.output_bias is not None:
            weights['output_bias'] = model.output_bias
        
        # Embedding weights
        emb_params = model.embedding.get_parameters()
        for key, value in emb_params.items():
            weights[f'embedding.{key}'] = value
        
        # Final layer norm weights
        final_norm_params = model.final_norm.get_parameters()
        for key, value in final_norm_params.items():
            weights[f'final_norm.{key}'] = value
        
        # All transformer layer weights
        for i, layer in enumerate(model.layers):
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                weights[f'layer_{i}.{key}'] = value
        
        return weights
    
    @staticmethod
    def _load_weights(model, weights) -> None:
        """Load weights into model"""
        # Load output projection
        if 'output_projection' in weights:
            model.output_projection = weights['output_projection']
        if 'output_bias' in weights:
            model.output_bias = weights['output_bias']
        
        # Load embedding weights
        emb_params = {k.replace('embedding.', ''): v 
                     for k, v in weights.items() 
                     if k.startswith('embedding.')}
        if emb_params:
            model.embedding.set_parameters(emb_params)
        
        # Load final layer norm
        final_norm_params = {k.replace('final_norm.', ''): v 
                           for k, v in weights.items() 
                           if k.startswith('final_norm.')}
        if final_norm_params:
            model.final_norm.set_parameters(final_norm_params)
        
        # Load transformer layers
        for i, layer in enumerate(model.layers):
            layer_params = {k.replace(f'layer_{i}.', ''): v 
                          for k, v in weights.items() 
                          if k.startswith(f'layer_{i}.')}
            if layer_params:
                layer.set_parameters(layer_params)
    
    @staticmethod
    def _verify_config(config1: dict, config2: dict) -> bool:
        """Verify two configurations match"""
        important_keys = ['vocab_size', 'embed_dim', 'num_heads', 
                         'num_layers', 'ff_dim', 'max_seq_len']
        
        for key in important_keys:
            if config1.get(key) != config2.get(key):
                print(f"  Config mismatch: {key} - {config1.get(key)} vs {config2.get(key)}")
                return False
        return True
    
    @staticmethod
    def save_training_state(save_dir: str, epoch: int, step: int, 
                           optimizer_state: Optional[dict] = None,
                           metrics: Optional[dict] = None) -> None:
        """Save training state for resuming"""
        save_path = Path(save_dir)
        state = {
            'epoch': epoch,
            'step': step,
            'optimizer_state': optimizer_state,
            'metrics': metrics
        }
        
        state_path = save_path / "training_state.json"
        with open(state_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json.dump(convert_numpy(state), f, indent=2)
        
        print(f"   - Training state: training_state.json")
    
    @staticmethod
    def list_checkpoints(directory: str) -> list:
        """List all available checkpoints in a directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        
        # Find all config files
        config_files = list(dir_path.glob("*_config.json"))
        checkpoints = []
        
        for config_file in config_files:
            name = config_file.stem.replace("_config", "")
            weights_file = dir_path / f"{name}_weights.npz"
            
            if weights_file.exists():
                # Get file info
                config_size = config_file.stat().st_size / 1024  # KB
                weights_size = weights_file.stat().st_size / (1024 * 1024)  # MB
                
                checkpoints.append({
                    'name': name,
                    'config': str(config_file),
                    'weights': str(weights_file),
                    'config_size_kb': f"{config_size:.1f}",
                    'weights_size_mb': f"{weights_size:.2f}"
                })
        
        return checkpoints


class ModelBuilder:
    """Build model from configuration file"""
    
    @staticmethod
    def from_config(config_path: str):
        """Create a model instance from a config file"""
        from transformer import Transformer
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = Transformer(config)
        print(f"Model built from config: {config_path}")
        return model
    
    @staticmethod
    def from_checkpoint(checkpoint_dir: str, name: str = "model"):
        """Create and load a model from checkpoint"""
        from transformer import Transformer
        
        checkpoint_path = Path(checkpoint_dir)
        config_path = checkpoint_path / f"{name}_config.json"
        
        # Build model from config
        model = ModelBuilder.from_config(config_path)
        
        # Load weights
        CheckpointManager.load(model, checkpoint_dir, name)
        
        return model


if __name__ == "__main__":
    print("Checkpoint Management System")
    print("-" * 40)
    print("\nThis module provides clean separation of:")
    print("  - Model configuration (architecture)")
    print("  - Model weights (parameters)")
    print("  - Training state (optional)")
    print("\nUsage:")
    print("  from checkpoint import CheckpointManager")
    print("  CheckpointManager.save(model, './checkpoints', 'my_model')")
    print("  CheckpointManager.load(model, './checkpoints', 'my_model')")
    print("\nOr build from checkpoint:")
    print("  from checkpoint import ModelBuilder")
    print("  model = ModelBuilder.from_checkpoint('./checkpoints', 'my_model')")