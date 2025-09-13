"""
Centralized model configuration for Pure Python Transformer

This file contains the single source of truth for model configurations
to ensure consistency across training and inference scripts.

Author: Joel Oliver
"""

# Default small model configuration
SMALL_MODEL_CONFIG = {
    'vocab_size': 1000,  # Will be updated based on tokenizer
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 2,
    'ff_dim': 256,
    'max_seq_len': 64,
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'ff_dropout': 0.0
}

# Improved larger model configuration
IMPROVED_MODEL_CONFIG = {
    'vocab_size': 1000,  # Will be updated based on tokenizer
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 4,
    'ff_dim': 512,
    'max_seq_len': 128,
    'dropout': 0.1,           # For training
    'attention_dropout': 0.1,  # For training
    'ff_dropout': 0.1         # For training
}

# Training configuration
DEFAULT_TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_epochs': 200,
    'seq_len': 64,
    'log_interval': 10,
    'warmup_steps': 100,
    'lr_decay': 0.95,
    'min_lr': 1e-5
}

# Quick test training configuration
TEST_TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_epochs': 20,
    'seq_len': 64,
    'log_interval': 10,
    'warmup_steps': 100,
    'lr_decay': 0.95,
    'min_lr': 1e-5
}

# BPE-specific training config (subword tokens allow smaller seq_len)
BPE_TRAINING_CONFIG = {
    'learning_rate': 2e-3,  # Can use higher LR with subwords
    'batch_size': 32,       # Larger batches with subwords
    'num_epochs': 50,       # Less epochs needed with subwords
    'seq_len': 32,          # Shorter sequences since subwords are more meaningful
    'log_interval': 10,
    'warmup_steps': 50,     # Less warmup needed
    'lr_decay': 0.98,       # Slower decay
    'min_lr': 1e-5
}

def get_model_config(config_name='improved'):
    """
    Get model configuration by name

    Args:
        config_name: 'small' or 'improved'

    Returns:
        Dictionary with model configuration
    """
    configs = {
        'small': SMALL_MODEL_CONFIG,
        'improved': IMPROVED_MODEL_CONFIG
    }
    return configs.get(config_name, IMPROVED_MODEL_CONFIG).copy()

def get_inference_config(config_name='improved'):
    """
    Get model configuration for inference (with dropout disabled)

    Args:
        config_name: 'small' or 'improved'

    Returns:
        Dictionary with model configuration for inference
    """
    config = get_model_config(config_name)
    # Disable dropout for inference
    config['dropout'] = 0.0
    config['attention_dropout'] = 0.0
    config['ff_dropout'] = 0.0
    return config