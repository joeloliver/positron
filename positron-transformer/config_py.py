"""
Configuration settings for Pure Python Transformer Implementation

This module contains all configuration settings for the transformer model,
training parameters, and numerical precision settings. Based on extensive
neural network experience from FPGA MLP implementation work.

Author: Joel Oliver
Based on: Master's thesis on FPGA Neural Network Implementation
"""

import os
from typing import Dict, Any, Optional
import numpy as np

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================

MODEL_CONFIG: Dict[str, Any] = {
    # Core architecture parameters
    'vocab_size': 8000,          # Size of vocabulary (start small for learning)
    'embed_dim': 256,            # Embedding dimension (d_model in paper)
    'num_heads': 8,              # Number of attention heads (must divide embed_dim)
    'num_layers': 6,             # Number of transformer layers
    'ff_dim': 1024,              # Feed-forward network hidden dimension
    'max_seq_len': 512,          # Maximum sequence length for positional encoding
    
    # Regularization parameters
    'dropout': 0.1,              # Dropout probability (0.0 to disable)
    'attention_dropout': 0.1,    # Dropout in attention mechanism
    'ff_dropout': 0.1,           # Dropout in feed-forward network
    
    # Architecture choices
    'use_bias': True,            # Whether to use bias in linear layers
    'layer_norm_eps': 1e-5,      # Epsilon for layer normalization
    'activation': 'gelu',        # Activation function: 'relu', 'gelu', 'swish'
    
    # Attention mechanism settings
    'attention_scale': None,     # Attention scaling factor (None for 1/sqrt(d_k))
    'causal_mask': True,         # Use causal masking for autoregressive modeling
}

# Validate model configuration
assert MODEL_CONFIG['embed_dim'] % MODEL_CONFIG['num_heads'] == 0, \
    f"embed_dim ({MODEL_CONFIG['embed_dim']}) must be divisible by num_heads ({MODEL_CONFIG['num_heads']})"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG: Dict[str, Any] = {
    # Optimization parameters
    'learning_rate': 1e-4,       # Initial learning rate
    'min_lr': 1e-6,              # Minimum learning rate for scheduling
    'weight_decay': 0.01,        # L2 regularization weight decay
    'beta1': 0.9,                # Adam optimizer beta1 parameter
    'beta2': 0.999,              # Adam optimizer beta2 parameter
    'epsilon': 1e-8,             # Adam optimizer epsilon
    
    # Training dynamics
    'batch_size': 32,            # Training batch size
    'eval_batch_size': 64,       # Evaluation batch size (can be larger)
    'num_epochs': 100,           # Total number of training epochs
    'max_steps': None,           # Maximum training steps (overrides epochs if set)
    
    # Learning rate scheduling
    'warmup_steps': 1000,        # Number of warmup steps
    'lr_schedule': 'cosine',     # Learning rate schedule: 'constant', 'cosine', 'linear'
    'warmup_init_lr': 1e-7,      # Initial learning rate during warmup
    
    # Gradient handling (crucial for training stability)
    'grad_clip': 1.0,            # Gradient clipping threshold
    'grad_clip_type': 'norm',    # Clipping type: 'norm', 'value'
    'accumulate_steps': 1,       # Gradient accumulation steps
    
    # Evaluation and logging
    'eval_interval': 500,        # Steps between evaluations
    'log_interval': 100,         # Steps between logging
    'save_interval': 1000,       # Steps between saving checkpoints
    'eval_steps': 100,           # Number of evaluation steps per evaluation
    
    # Early stopping
    'patience': 10,              # Early stopping patience (epochs)
    'min_improvement': 1e-4,     # Minimum improvement for early stopping
}

# =============================================================================
# NUMERICAL PRECISION CONFIGURATION
# =============================================================================
# Based on FPGA floating-point implementation experience

PRECISION_CONFIG: Dict[str, Any] = {
    # Data types (leveraging IEEE 754 expertise from FPGA work)
    'dtype': np.float32,         # Primary data type (single precision)
    'compute_dtype': np.float32, # Computation data type
    'param_dtype': np.float32,   # Parameter storage data type
    
    # Numerical stability parameters
    'eps': 1e-8,                 # Small epsilon for numerical stability
    'softmax_temp': 1.0,         # Temperature for softmax (for numerical stability)
    'layer_norm_eps': 1e-5,      # Layer normalization epsilon
    
    # Initialization parameters (critical for training stability)
    'init_std': 0.02,            # Standard deviation for normal initialization
    'init_gain': 1.0,            # Gain for Xavier/Glorot initialization
    'embedding_init_std': 0.02,  # Standard deviation for embedding initialization
    
    # Activation function parameters
    'gelu_approximate': False,   # Use approximate GELU (faster but less accurate)
    'swish_beta': 1.0,          # Beta parameter for Swish activation
    
    # Attention mechanism precision
    'attention_softmax_dtype': np.float32,  # Data type for attention softmax
    'qkv_bias': True,           # Use bias in Q, K, V projections
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG: Dict[str, Any] = {
    # Tokenization settings
    'tokenizer_type': 'character',  # 'character', 'word', 'bpe'
    'vocab_file': None,             # Path to vocabulary file (optional)
    'max_vocab_size': 8000,         # Maximum vocabulary size
    'min_token_freq': 2,            # Minimum token frequency for inclusion
    
    # Special tokens
    'pad_token': '<PAD>',           # Padding token
    'unk_token': '<UNK>',           # Unknown token
    'bos_token': '<BOS>',           # Beginning of sequence token
    'eos_token': '<EOS>',           # End of sequence token
    
    # Data processing
    'max_length': 512,              # Maximum sequence length
    'truncation': True,             # Truncate sequences longer than max_length
    'padding': 'max_length',        # Padding strategy
    
    # Data loading
    'shuffle_buffer': 10000,        # Buffer size for shuffling
    'num_workers': 0,               # Number of data loading workers (0 for single-threaded)
    'pin_memory': False,            # Pin memory for faster GPU transfer
}

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS: Dict[str, str] = {
    # Data directories
    'data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'train_data': os.path.join(PROJECT_ROOT, 'data', 'train'),
    'eval_data': os.path.join(PROJECT_ROOT, 'data', 'eval'),
    'vocab_dir': os.path.join(PROJECT_ROOT, 'data', 'vocab'),
    
    # Model and checkpoint directories
    'checkpoint_dir': os.path.join(PROJECT_ROOT, 'checkpoints'),
    'model_dir': os.path.join(PROJECT_ROOT, 'models'),
    'cache_dir': os.path.join(PROJECT_ROOT, 'cache'),
    
    # Logging and output directories
    'log_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'output_dir': os.path.join(PROJECT_ROOT, 'output'),
    'viz_dir': os.path.join(PROJECT_ROOT, 'visualizations'),
    
    # Documentation
    'docs_dir': os.path.join(PROJECT_ROOT, 'docs'),
}

# Create only essential directories by default
essential_dirs = ['data_dir', 'checkpoint_dir']
for key in essential_dirs:
    if key in PATHS:
        os.makedirs(PATHS[key], exist_ok=True)

# Function to create all directories if needed
def create_all_directories():
    """Create all configured directories."""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

# =============================================================================
# DEVICE AND PERFORMANCE CONFIGURATION
# =============================================================================

DEVICE_CONFIG: Dict[str, Any] = {
    # Computation settings
    'device': 'cpu',             # Device for computation ('cpu', 'cuda' if available)
    'seed': 42,                  # Random seed for reproducibility
    'deterministic': True,       # Use deterministic algorithms
    
    # Performance settings
    'compile_model': False,      # Compile model for faster inference (if supported)
    'use_amp': False,           # Use automatic mixed precision (if supported)
    'benchmark': False,          # Enable benchmarking mode
    
    # Memory management
    'max_memory_gb': None,       # Maximum memory usage in GB (None for unlimited)
    'empty_cache_freq': 100,     # Frequency to empty cache (steps)
}

# =============================================================================
# GENERATION CONFIGURATION
# =============================================================================

GENERATION_CONFIG: Dict[str, Any] = {
    # Generation parameters
    'max_length': 100,           # Maximum generation length
    'temperature': 1.0,          # Sampling temperature
    'top_k': 50,                 # Top-k sampling
    'top_p': 0.9,                # Nucleus sampling (top-p)
    'repetition_penalty': 1.0,   # Repetition penalty
    
    # Generation strategies
    'do_sample': True,           # Use sampling instead of greedy decoding
    'num_beams': 1,              # Number of beams for beam search (1 for no beam search)
    'early_stopping': True,      # Stop generation at EOS token
    
    # Special token handling
    'pad_token_id': None,        # Will be set by tokenizer
    'bos_token_id': None,        # Will be set by tokenizer
    'eos_token_id': None,        # Will be set by tokenizer
}

# =============================================================================
# DEBUGGING AND DEVELOPMENT CONFIGURATION
# =============================================================================

DEBUG_CONFIG: Dict[str, Any] = {
    # Debugging options
    'debug': False,              # Enable debug mode
    'verbose': True,             # Verbose logging
    'profile': False,            # Enable profiling
    'trace_memory': False,       # Trace memory usage
    
    # Validation options
    'check_gradients': False,    # Check gradients during training
    'validate_shapes': True,     # Validate tensor shapes
    'check_nan': True,           # Check for NaN values
    
    # Testing options
    'overfit_small_batch': False, # Overfit on small batch for testing
    'fast_dev_run': False,       # Run single batch for development
    'limit_train_batches': None, # Limit number of training batches
    'limit_eval_batches': None,  # Limit number of evaluation batches
}

# =============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =============================================================================

def validate_config() -> None:
    """
    Validate all configuration parameters for consistency and correctness.
    
    This function checks that all configuration parameters are valid and
    compatible with each other. Based on experience with hardware constraints
    and numerical precision from FPGA implementation work.
    """
    
    # Validate model architecture
    assert MODEL_CONFIG['embed_dim'] > 0, "embed_dim must be positive"
    assert MODEL_CONFIG['num_heads'] > 0, "num_heads must be positive"
    assert MODEL_CONFIG['embed_dim'] % MODEL_CONFIG['num_heads'] == 0, \
        "embed_dim must be divisible by num_heads"
    assert MODEL_CONFIG['num_layers'] > 0, "num_layers must be positive"
    assert MODEL_CONFIG['ff_dim'] > 0, "ff_dim must be positive"
    assert MODEL_CONFIG['max_seq_len'] > 0, "max_seq_len must be positive"
    
    # Validate training configuration
    assert TRAINING_CONFIG['learning_rate'] > 0, "learning_rate must be positive"
    assert TRAINING_CONFIG['batch_size'] > 0, "batch_size must be positive"
    assert TRAINING_CONFIG['num_epochs'] > 0, "num_epochs must be positive"
    assert 0 <= TRAINING_CONFIG['beta1'] < 1, "beta1 must be in [0, 1)"
    assert 0 <= TRAINING_CONFIG['beta2'] < 1, "beta2 must be in [0, 1)"
    assert TRAINING_CONFIG['epsilon'] > 0, "epsilon must be positive"
    
    # Validate dropout rates
    for key in ['dropout', 'attention_dropout', 'ff_dropout']:
        if key in MODEL_CONFIG:
            assert 0 <= MODEL_CONFIG[key] <= 1, f"{key} must be in [0, 1]"
    
    # Validate precision configuration
    assert PRECISION_CONFIG['init_std'] > 0, "init_std must be positive"
    assert PRECISION_CONFIG['eps'] > 0, "eps must be positive"
    
    print("✅ All configuration parameters validated successfully!")

def get_model_summary() -> Dict[str, Any]:
    """
    Get a summary of the model configuration for logging and debugging.
    
    Returns:
        Dict containing model summary information
    """
    
    # Calculate model parameters (approximate)
    vocab_size = MODEL_CONFIG['vocab_size']
    embed_dim = MODEL_CONFIG['embed_dim']
    num_layers = MODEL_CONFIG['num_layers']
    ff_dim = MODEL_CONFIG['ff_dim']
    
    # Approximate parameter count
    embedding_params = vocab_size * embed_dim
    attention_params = num_layers * (4 * embed_dim * embed_dim)  # Q, K, V, O projections
    ff_params = num_layers * (2 * embed_dim * ff_dim)  # Two linear layers
    layer_norm_params = num_layers * 2 * embed_dim  # Two layer norms per layer
    output_params = embed_dim * vocab_size  # Output projection
    
    total_params = (embedding_params + attention_params + ff_params + 
                   layer_norm_params + output_params)
    
    return {
        'model_name': 'Pure Python Transformer',
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'num_heads': MODEL_CONFIG['num_heads'],
        'num_layers': num_layers,
        'ff_dim': ff_dim,
        'max_seq_len': MODEL_CONFIG['max_seq_len'],
        'total_params_approx': total_params,
        'total_params_millions': total_params / 1e6,
        'memory_footprint_mb_approx': total_params * 4 / 1e6,  # 4 bytes per float32
        'dropout': MODEL_CONFIG['dropout'],
        'learning_rate': TRAINING_CONFIG['learning_rate'],
        'batch_size': TRAINING_CONFIG['batch_size'],
    }

def print_config_summary() -> None:
    """
    Print a formatted summary of all configurations.
    """
    print("=" * 80)
    print("PURE PYTHON TRANSFORMER CONFIGURATION SUMMARY")
    print("=" * 80)
    
    summary = get_model_summary()
    
    print(f"\n📐 MODEL ARCHITECTURE:")
    print(f"  • Model: {summary['model_name']}")
    print(f"  • Vocabulary Size: {summary['vocab_size']:,}")
    print(f"  • Embedding Dimension: {summary['embed_dim']}")
    print(f"  • Attention Heads: {summary['num_heads']}")
    print(f"  • Layers: {summary['num_layers']}")
    print(f"  • Feed-Forward Dimension: {summary['ff_dim']}")
    print(f"  • Max Sequence Length: {summary['max_seq_len']}")
    print(f"  • Approximate Parameters: {summary['total_params_millions']:.1f}M")
    print(f"  • Approximate Memory: {summary['memory_footprint_mb_approx']:.1f} MB")
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"  • Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  • Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  • Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  • Warmup Steps: {TRAINING_CONFIG['warmup_steps']}")
    print(f"  • Dropout: {MODEL_CONFIG['dropout']}")
    print(f"  • Weight Decay: {TRAINING_CONFIG['weight_decay']}")
    print(f"  • Gradient Clipping: {TRAINING_CONFIG['grad_clip']}")
    
    print(f"\n🔧 PRECISION CONFIGURATION:")
    print(f"  • Data Type: {PRECISION_CONFIG['dtype']}")
    print(f"  • Initialization Std: {PRECISION_CONFIG['init_std']}")
    print(f"  • Numerical Epsilon: {PRECISION_CONFIG['eps']}")
    print(f"  • Layer Norm Epsilon: {PRECISION_CONFIG['layer_norm_eps']}")
    
    print("=" * 80)

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print_config_summary()
else:
    validate_config()