"""
Mathematical utilities for Pure Python Transformer Implementation

This module contains all mathematical functions, activation functions, and
utility functions needed for transformer implementation. Based on extensive
experience with numerical precision from FPGA neural network work.

Key Features:
- Numerically stable implementations (leveraging IEEE 754 expertise)
- Efficient vectorized operations using NumPy
- Comprehensive activation function library
- Gradient computation utilities
- Initialization schemes

Author: Joel Oliver
Based on: Master's thesis on FPGA Neural Network Implementation
"""

import numpy as np
import math
from typing import Optional, Tuple, Union, Callable, Dict, Any
import warnings
from functools import wraps

# =============================================================================
# NUMERICAL STABILITY UTILITIES
# =============================================================================

def stable_softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax implementation.
    
    This implementation prevents overflow by subtracting the maximum value
    before exponentiation. Based on experience with floating-point arithmetic
    precision from FPGA implementation work.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
        temperature: Temperature parameter for scaling (default: 1.0)
    
    Returns:
        Softmax probabilities along specified axis
        
    Mathematical Formula:
        softmax(x_i) = exp(x_i / T) / sum(exp(x_j / T) for all j)
        
    Numerical Stability:
        softmax(x_i) = exp((x_i - max(x)) / T) / sum(exp((x_j - max(x)) / T))
    """
    # Scale by temperature
    x_scaled = x / temperature
    
    # Subtract maximum for numerical stability (prevents overflow)
    x_shifted = x_scaled - np.max(x_scaled, axis=axis, keepdims=True)
    
    # Compute exponentials
    exp_x = np.exp(x_shifted)
    
    # Normalize to get probabilities
    softmax_probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    return softmax_probs

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log-softmax implementation.
    
    More stable than computing log(softmax(x)) directly.
    
    Args:
        x: Input array
        axis: Axis along which to compute log-softmax
    
    Returns:
        Log-softmax values
    """
    # Shift for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Compute log-sum-exp
    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
    
    return x_shifted - log_sum_exp

def stable_log_sum_exp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log-sum-exp computation.
    
    Computes log(sum(exp(x))) in a numerically stable way.
    
    Args:
        x: Input array
        axis: Axis along which to compute
    
    Returns:
        Log-sum-exp values
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================
# Based on experience with activation function approximations from FPGA work

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x: Input array
    
    Returns:
        ReLU(x) = max(0, x)
    """
    return np.maximum(0.0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation function.
    
    Args:
        x: Input array
    
    Returns:
        Derivative: 1 if x > 0, 0 otherwise
    """
    return (x > 0).astype(x.dtype)

def gelu(x: np.ndarray, approximate: bool = False) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU is commonly used in transformer models like BERT and GPT.
    
    Args:
        x: Input array
        approximate: Use approximate computation (faster but less accurate)
    
    Returns:
        GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
                  of the standard normal distribution
    
    Exact formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    Approximate: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    if approximate:
        # Approximate GELU (faster computation)
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + np.tanh(inner))
    else:
        # Exact GELU using error function
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5 * x * (1.0 + np.vectorize(lambda x: math.erf(x / np.sqrt(2.0)))(x))

def gelu_derivative(x: np.ndarray, approximate: bool = False) -> np.ndarray:
    """
    Derivative of GELU activation function.
    
    Args:
        x: Input array
        approximate: Use approximate computation
    
    Returns:
        Derivative of GELU
    """
    if approximate:
        # Approximate derivative
        tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech_squared = 1.0 - tanh_val**2
        
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech_squared * np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x**2)
    else:
        # Exact derivative using error function and Gaussian
        erf_val = np.vectorize(lambda x: math.erf(x / np.sqrt(2.0)))(x)
        gaussian = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
        return 0.5 * (1.0 + erf_val) + x * gaussian

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish activation function (also known as SiLU - Sigmoid Linear Unit).
    
    Swish(x) = x * sigmoid(β * x)
    
    Args:
        x: Input array
        beta: Scaling parameter (default: 1.0)
    
    Returns:
        Swish activation
    """
    return x * sigmoid(beta * x)

def swish_derivative(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Derivative of Swish activation function.
    
    Args:
        x: Input array
        beta: Scaling parameter
    
    Returns:
        Derivative of Swish
    """
    sig = sigmoid(beta * x)
    return sig + x * sig * (1.0 - sig) * beta

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function with numerical stability.
    
    Based on experience with sigmoid approximations from FPGA work.
    This implementation handles large positive and negative values to
    prevent overflow/underflow.
    
    Args:
        x: Input array
    
    Returns:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    # Clamp extreme values to prevent overflow
    x_clamped = np.clip(x, -500, 500)
    
    # Use numerically stable implementation
    positive_mask = x_clamped >= 0
    
    # For positive values: sigmoid(x) = 1 / (1 + exp(-x))
    # For negative values: sigmoid(x) = exp(x) / (1 + exp(x))
    result = np.zeros_like(x_clamped)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-x_clamped[positive_mask]))
    result[~positive_mask] = np.exp(x_clamped[~positive_mask]) / (1.0 + np.exp(x_clamped[~positive_mask]))
    
    return result

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation function.
    
    Args:
        x: Input array
    
    Returns:
        Derivative: sigmoid(x) * (1 - sigmoid(x))
    """
    sig = sigmoid(x)
    return sig * (1.0 - sig)

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.
    
    Args:
        x: Input array
    
    Returns:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh activation function.
    
    Args:
        x: Input array
    
    Returns:
        Derivative: 1 - tanh²(x)
    """
    tanh_x = np.tanh(x)
    return 1.0 - tanh_x**2

# Dictionary mapping activation names to functions
ACTIVATION_FUNCTIONS = {
    'relu': relu,
    'gelu': gelu,
    'swish': swish,
    'sigmoid': sigmoid,
    'tanh': tanh,
}

ACTIVATION_DERIVATIVES = {
    'relu': relu_derivative,
    'gelu': gelu_derivative,
    'swish': swish_derivative,
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
}

def get_activation_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
    
    Returns:
        Activation function
    
    Raises:
        ValueError: If activation function not found
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}. "
                        f"Available: {list(ACTIVATION_FUNCTIONS.keys())}")
    return ACTIVATION_FUNCTIONS[name]

def get_activation_derivative(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get activation function derivative by name.
    
    Args:
        name: Name of activation function
    
    Returns:
        Activation function derivative
    
    Raises:
        ValueError: If activation function derivative not found
    """
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation derivative: {name}. "
                        f"Available: {list(ACTIVATION_DERIVATIVES.keys())}")
    return ACTIVATION_DERIVATIVES[name]

# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================
# Based on proven initialization schemes and FPGA implementation experience

def xavier_uniform_init(shape: Tuple[int, ...], gain: float = 1.0, 
                       dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.
    
    Weights are sampled from uniform distribution:
    U(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor
        dtype: Data type
    
    Returns:
        Initialized weight array
    """
    if len(shape) < 2:
        raise ValueError("Xavier initialization requires at least 2D tensor")
    
    fan_in = shape[-2]
    fan_out = shape[-1]
    
    # Calculate bound
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    
    return np.random.uniform(-bound, bound, shape).astype(dtype)

def xavier_normal_init(shape: Tuple[int, ...], gain: float = 1.0,
                      dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Xavier/Glorot normal initialization.
    
    Weights are sampled from normal distribution:
    N(0, gain * sqrt(2 / (fan_in + fan_out)))
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor
        dtype: Data type
    
    Returns:
        Initialized weight array
    """
    if len(shape) < 2:
        raise ValueError("Xavier initialization requires at least 2D tensor")
    
    fan_in = shape[-2]
    fan_out = shape[-1]
    
    # Calculate standard deviation
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    
    return np.random.normal(0.0, std, shape).astype(dtype)

def kaiming_uniform_init(shape: Tuple[int, ...], gain: float = 1.0, 
                        mode: str = 'fan_in', dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Kaiming/He uniform initialization.
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor
        mode: Either 'fan_in' or 'fan_out'
        dtype: Data type
    
    Returns:
        Initialized weight array
    """
    if len(shape) < 2:
        raise ValueError("Kaiming initialization requires at least 2D tensor")
    
    if mode == 'fan_in':
        fan = shape[-2]
    elif mode == 'fan_out':
        fan = shape[-1]
    else:
        raise ValueError("Mode must be either 'fan_in' or 'fan_out'")
    
    bound = gain * np.sqrt(6.0 / fan)
    
    return np.random.uniform(-bound, bound, shape).astype(dtype)

def kaiming_normal_init(shape: Tuple[int, ...], gain: float = 1.0,
                       mode: str = 'fan_in', dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Kaiming/He normal initialization.
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor
        mode: Either 'fan_in' or 'fan_out'
        dtype: Data type
    
    Returns:
        Initialized weight array
    """
    if len(shape) < 2:
        raise ValueError("Kaiming initialization requires at least 2D tensor")
    
    if mode == 'fan_in':
        fan = shape[-2]
    elif mode == 'fan_out':
        fan = shape[-1]
    else:
        raise ValueError("Mode must be either 'fan_in' or 'fan_out'")
    
    std = gain * np.sqrt(2.0 / fan)
    
    return np.random.normal(0.0, std, shape).astype(dtype)

def normal_init(shape: Tuple[int, ...], mean: float = 0.0, std: float = 0.02,
               dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Normal initialization with specified mean and standard deviation.
    
    Args:
        shape: Shape of weight tensor
        mean: Mean of normal distribution
        std: Standard deviation
        dtype: Data type
    
    Returns:
        Initialized weight array
    """
    return np.random.normal(mean, std, shape).astype(dtype)

def zeros_init(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Zero initialization.
    
    Args:
        shape: Shape of tensor
        dtype: Data type
    
    Returns:
        Zero-initialized array
    """
    return np.zeros(shape, dtype=dtype)

def ones_init(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Ones initialization.
    
    Args:
        shape: Shape of tensor
        dtype: Data type
    
    Returns:
        Ones-initialized array
    """
    return np.ones(shape, dtype=dtype)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_array(x: Union[np.ndarray, float, int]) -> np.ndarray:
    """
    Ensure input is a NumPy array.
    
    Args:
        x: Input value or array
    
    Returns:
        NumPy array
    """
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x

def check_finite(x: np.ndarray, name: str = "array") -> None:
    """
    Check that all values in array are finite (no NaN or Inf).
    
    Args:
        x: Array to check
        name: Name for error message
    
    Raises:
        ValueError: If array contains non-finite values
    """
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

def compute_cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute cross-entropy loss for language modeling
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: True labels of shape (batch_size, seq_len)
        
    Returns:
        Average cross-entropy loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for easier processing
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    
    # Compute log softmax
    log_probs = log_softmax(logits_flat, axis=-1)
    
    # Gather log probabilities for true labels
    batch_indices = np.arange(len(labels_flat))
    selected_log_probs = log_probs[batch_indices, labels_flat]
    
    # Average negative log likelihood
    loss = -np.mean(selected_log_probs)
    
    return loss

def clip_gradients(gradients: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
    """
    Clip gradients by global norm
    
    Args:
        gradients: Dictionary of gradients
        max_norm: Maximum gradient norm
        
    Returns:
        Dictionary of clipped gradients
    """
    # Compute global norm
    total_norm = 0.0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        scale_factor = max_norm / total_norm
        clipped_gradients = {k: v * scale_factor for k, v in gradients.items()}
    else:
        clipped_gradients = gradients
    
    return clipped_gradients

def gradient_clipping(gradients: np.ndarray, max_norm: float, 
                     norm_type: str = 'l2') -> Tuple[np.ndarray, float]:
    """
    Clip gradients by norm to prevent gradient explosion.
    
    Args:
        gradients: Gradient array
        max_norm: Maximum allowed norm
        norm_type: Type of norm ('l2' or 'l1')
    
    Returns:
        Tuple of (clipped_gradients, actual_norm)
    """
    if norm_type == 'l2':
        norm = np.linalg.norm(gradients)
    elif norm_type == 'l1':
        norm = np.sum(np.abs(gradients))
    else:
        raise ValueError("norm_type must be 'l2' or 'l1'")
    
    if norm > max_norm:
        clipped_gradients = gradients * (max_norm / norm)
    else:
        clipped_gradients = gradients
    
    return clipped_gradients, norm

def create_causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Create causal (lower triangular) mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        dtype: Data type
    
    Returns:
        Causal mask of shape (seq_len, seq_len)
        Values: 0 for allowed positions, -inf for masked positions
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask.astype(dtype)
    mask[mask == 1] = -np.inf
    return mask

def create_padding_mask(lengths: np.ndarray, max_len: int, 
                       dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Create padding mask for variable length sequences.
    
    Args:
        lengths: Array of sequence lengths
        max_len: Maximum sequence length
        dtype: Data type
    
    Returns:
        Padding mask of shape (batch_size, max_len)
        Values: 0 for valid positions, -inf for padded positions
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len), dtype=dtype)
    
    for i, length in enumerate(lengths):
        if length < max_len:
            mask[i, length:] = -np.inf
    
    return mask

def compute_accuracy(predictions: np.ndarray, targets: np.ndarray, 
                    top_k: int = 1) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Predicted logits of shape (batch_size, num_classes)
        targets: Target labels of shape (batch_size,)
        top_k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as float between 0 and 1
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if targets.ndim == 0:
        targets = np.array([targets])
    
    # Get top-k predictions
    top_k_preds = np.argsort(predictions, axis=-1)[:, -top_k:]
    
    # Check if targets are in top-k predictions
    correct = np.any(top_k_preds == targets.reshape(-1, 1), axis=1)
    
    return np.mean(correct)

def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)

@wraps(np.ndarray)
def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
               eps: float = 1e-8) -> np.ndarray:
    """
    Safe division that prevents division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Small epsilon to add to denominator
    
    Returns:
        Result of safe division
    """
    return numerator / (denominator + eps)

# =============================================================================
# DEBUGGING AND PROFILING UTILITIES
# =============================================================================

def print_array_stats(x: np.ndarray, name: str = "array") -> None:
    """
    Print statistics about an array for debugging.
    
    Args:
        x: Array to analyze
        name: Name for display
    """
    print(f"\n{name} statistics:")
    print(f"  Shape: {x.shape}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Min: {np.min(x):.6f}")
    print(f"  Max: {np.max(x):.6f}")
    print(f"  Mean: {np.mean(x):.6f}")
    print(f"  Std: {np.std(x):.6f}")
    print(f"  Finite values: {np.sum(np.isfinite(x))}/{x.size}")
    if not np.all(np.isfinite(x)):
        print(f"  NaN values: {np.sum(np.isnan(x))}")
        print(f"  Inf values: {np.sum(np.isinf(x))}")

def check_gradient_finite(gradient: np.ndarray, parameter_name: str) -> bool:
    """
    Check if gradients are finite and warn if not.
    
    Args:
        gradient: Gradient array
        parameter_name: Name of parameter for warning
    
    Returns:
        True if gradients are finite, False otherwise
    """
    if not np.all(np.isfinite(gradient)):
        warnings.warn(f"Non-finite gradients detected in {parameter_name}!")
        return False
    return True