"""
Layer Normalization for Pure Python Transformer Implementation

This module implements layer normalization, a crucial component for training
stability in transformer models. Based on experience with numerical stability
and normalization techniques from FPGA neural network implementation.

Layer normalization normalizes inputs across the feature dimension (last axis)
rather than the batch dimension like batch normalization. This makes it
particularly suitable for sequence models where batch statistics may vary.

Mathematical Formula:
    LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
    
    where:
    μ = mean(x, axis=-1, keepdims=True)  # Mean across features
    σ = std(x, axis=-1, keepdims=True)   # Standard deviation across features
    γ = learnable scale parameter (initialized to 1)
    β = learnable bias parameter (initialized to 0)
    ε = small constant for numerical stability

Author: Joel Oliver
Based on: Master's thesis on FPGA Neural Network Implementation
"""

import numpy as np
from typing import Optional, Dict, Tuple, Union

from utils_py import ones_init, zeros_init, check_finite, print_array_stats
from config_py import PRECISION_CONFIG

# =============================================================================
# LAYER NORMALIZATION
# =============================================================================

class LayerNorm:
    """
    Layer Normalization implementation.
    
    Normalizes inputs across the feature dimension, providing training stability
    and faster convergence. Unlike batch normalization, layer norm doesn't
    depend on batch statistics, making it more suitable for transformers.
    
    This implementation handles:
    - Numerical stability (avoiding division by zero)
    - Proper gradient computation
    - Configurable epsilon for different precision requirements
    - Support for different data types
    """
    
    def __init__(self,
                 normalized_shape: Union[int, Tuple[int, ...]],
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 dtype: np.dtype = np.float32):
        """
        Initialize layer normalization.
        
        Args:
            normalized_shape: Shape of normalized axes (typically embed_dim)
            eps: Small constant for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
            dtype: Data type for parameters
        """
        # Handle both int and tuple inputs
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape
        
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype
        
        # Initialize learnable parameters if enabled
        if self.elementwise_affine:
            # Scale parameter (γ) - initialized to 1
            self.weight = ones_init(self.normalized_shape, dtype=dtype)
            # Bias parameter (β) - initialized to 0
            self.bias = zeros_init(self.normalized_shape, dtype=dtype)
            
            # Gradient storage
            self.weight_grad = np.zeros_like(self.weight)
            self.bias_grad = np.zeros_like(self.bias)
        else:
            self.weight = None
            self.bias = None
            self.weight_grad = None
            self.bias_grad = None
        
        # Cache for backward pass
        self._cache = {}
        
        print(f"LayerNorm initialized: normalized_shape={self.normalized_shape}, "
              f"eps={eps}, elementwise_affine={elementwise_affine}")
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving"""
        params = {}
        if self.elementwise_affine:
            params['weight'] = self.weight
            params['bias'] = self.bias
        return params
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from loaded checkpoint"""
        if self.elementwise_affine:
            self.weight = params['weight']
            self.bias = params['bias']
            self.weight_grad = np.zeros_like(self.weight)
            self.bias_grad = np.zeros_like(self.bias)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through layer normalization.
        
        Args:
            x: Input tensor of shape (..., *normalized_shape)
            
        Returns:
            Normalized output of same shape as input
        """
        # Validate input shape
        input_shape = x.shape
        if input_shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f"Input shape {input_shape} doesn't match "
                           f"normalized_shape {self.normalized_shape}")
        
        # Compute normalization axes (all axes corresponding to normalized_shape)
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))  # Last ndim axes
        
        # Compute mean and variance
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Apply affine transformation if enabled
        if self.elementwise_affine:
            output = self.weight * x_norm + self.bias
        else:
            output = x_norm
        
        # Cache values for backward pass
        self._cache = {
            'x': x,
            'mean': mean,
            'var': var,
            'x_norm': x_norm,
            'axes': axes
        }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through layer normalization.
        
        This implements the gradients with respect to input and parameters.
        The mathematics involves careful handling of the normalization
        operation's dependencies.
        
        Args:
            grad_output: Gradient from upstream layers
            
        Returns:
            Gradient with respect to input
        """
        if not self._cache:
            raise RuntimeError("Must call forward before backward")
        
        # Retrieve cached values
        x = self._cache['x']
        mean = self._cache['mean']
        var = self._cache['var']
        x_norm = self._cache['x_norm']
        axes = self._cache['axes']
        
        # Get normalization dimensions
        N = 1
        for axis in axes:
            N *= x.shape[axis]
        
        # Compute gradients with respect to parameters
        if self.elementwise_affine:
            # Gradient w.r.t. weight (scale parameter)
            self.weight_grad += np.sum(grad_output * x_norm, 
                                     axis=tuple(range(len(grad_output.shape) - len(self.normalized_shape))))
            
            # Gradient w.r.t. bias
            self.bias_grad += np.sum(grad_output,
                                   axis=tuple(range(len(grad_output.shape) - len(self.normalized_shape))))
            
            # Scale gradient by weight for input gradient computation
            grad_output = grad_output * self.weight
        
        # Compute gradient with respect to input
        # This is the most complex part - derivatives of normalization
        
        # Standard deviation
        std = np.sqrt(var + self.eps)
        
        # Gradient components
        # d_x_norm = grad_output (already scaled by weight if applicable)
        d_x_norm = grad_output
        
        # d_var = sum(d_x_norm * (x - mean)) * (-0.5) * (var + eps)^(-3/2)
        d_var = np.sum(d_x_norm * (x - mean), axis=axes, keepdims=True) * (-0.5) * (var + self.eps)**(-1.5)
        
        # d_mean = sum(d_x_norm * (-1/std)) + d_var * sum(-2*(x-mean)) / N
        d_mean = (np.sum(d_x_norm * (-1.0 / std), axis=axes, keepdims=True) + 
                 d_var * np.sum(-2.0 * (x - mean), axis=axes, keepdims=True) / N)
        
        # d_x = d_x_norm/std + d_var*2*(x-mean)/N + d_mean/N
        grad_input = (d_x_norm / std + 
                     d_var * 2.0 * (x - mean) / N + 
                     d_mean / N)
        
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters dictionary."""
        if self.elementwise_affine:
            return {
                'weight': self.weight,
                'bias': self.bias
            }
        return {}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients dictionary."""
        if self.elementwise_affine:
            return {
                'weight': self.weight_grad,
                'bias': self.bias_grad
            }
        return {}
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        if self.elementwise_affine:
            self.weight_grad.fill(0.0)
            self.bias_grad.fill(0.0)

# =============================================================================
# ROOT MEAN SQUARE LAYER NORMALIZATION (RMSNorm)
# =============================================================================

class RMSNorm:
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    A simplified version of LayerNorm that only normalizes by RMS without
    centering (no mean subtraction). This is used in some modern architectures
    like LLaMA and can be more stable in certain cases.
    
    Mathematical Formula:
        RMSNorm(x) = γ * x / RMS(x)
        where RMS(x) = sqrt(mean(x²) + ε)
    
    This version is computationally simpler and often performs similarly
    to full LayerNorm while being more efficient.
    """
    
    def __init__(self,
                 normalized_shape: Union[int, Tuple[int, ...]],
                 eps: float = 1e-8,
                 elementwise_affine: bool = True,
                 dtype: np.dtype = np.float32):
        """
        Initialize RMS normalization.
        
        Args:
            normalized_shape: Shape of normalized axes
            eps: Small constant for numerical stability
            elementwise_affine: Whether to use learnable scale parameter
            dtype: Data type for parameters
        """
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape
        
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype
        
        # Initialize scale parameter if enabled
        if self.elementwise_affine:
            self.weight = ones_init(self.normalized_shape, dtype=dtype)
            self.weight_grad = np.zeros_like(self.weight)
        else:
            self.weight = None
            self.weight_grad = None
        
        # Cache for backward pass
        self._cache = {}
        
        print(f"RMSNorm initialized: normalized_shape={self.normalized_shape}, "
              f"eps={eps}, elementwise_affine={elementwise_affine}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            RMS normalized output
        """
        # Validate input shape
        input_shape = x.shape
        if input_shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f"Input shape {input_shape} doesn't match "
                           f"normalized_shape {self.normalized_shape}")
        
        # Compute normalization axes
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))
        
        # Compute RMS
        ms = np.mean(x**2, axis=axes, keepdims=True)  # Mean square
        rms = np.sqrt(ms + self.eps)  # Root mean square
        
        # Normalize
        x_norm = x / rms
        
        # Apply scale if enabled
        if self.elementwise_affine:
            output = self.weight * x_norm
        else:
            output = x_norm
        
        # Cache for backward pass
        self._cache = {
            'x': x,
            'rms': rms,
            'x_norm': x_norm,
            'axes': axes
        }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through RMS normalization.
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient with respect to input
        """
        if not self._cache:
            raise RuntimeError("Must call forward before backward")
        
        # Retrieve cached values
        x = self._cache['x']
        rms = self._cache['rms']
        x_norm = self._cache['x_norm']
        axes = self._cache['axes']
        
        # Get normalization dimensions
        N = 1
        for axis in axes:
            N *= x.shape[axis]
        
        # Compute gradient with respect to weight
        if self.elementwise_affine:
            self.weight_grad += np.sum(grad_output * x_norm,
                                     axis=tuple(range(len(grad_output.shape) - len(self.normalized_shape))))
            # Scale gradient by weight
            grad_output = grad_output * self.weight
        
        # Compute gradient with respect to input
        # d_rms = sum(grad_output * x * (-1/rms²))
        d_rms = np.sum(grad_output * x * (-1.0 / rms**2), axis=axes, keepdims=True)
        
        # d_ms = d_rms * (1 / (2*sqrt(ms + eps)))
        ms = rms**2 - self.eps
        d_ms = d_rms * (1.0 / (2.0 * np.sqrt(ms + self.eps)))
        
        # d_x = grad_output/rms + d_ms * 2*x / N
        grad_input = grad_output / rms + d_ms * 2.0 * x / N
        
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters dictionary."""
        if self.elementwise_affine:
            return {'weight': self.weight}
        return {}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients dictionary."""
        if self.elementwise_affine:
            return {'weight': self.weight_grad}
        return {}
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        if self.elementwise_affine:
            self.weight_grad.fill(0.0)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_norm_layer(norm_type: str = 'layer_norm', 
                     normalized_shape: Union[int, Tuple[int, ...]] = None,
                     **kwargs) -> Union[LayerNorm, RMSNorm]:
    """
    Factory function to create normalization layers.
    
    Args:
        norm_type: Type of normalization ('layer_norm' or 'rms_norm')
        normalized_shape: Shape of normalized dimensions
        **kwargs: Additional arguments for the normalization layer
        
    Returns:
        Normalization layer instance
    """
    if norm_type == 'layer_norm':
        return LayerNorm(normalized_shape, **kwargs)
    elif norm_type == 'rms_norm':
        return RMSNorm(normalized_shape, **kwargs)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. "
                        f"Supported: ['layer_norm', 'rms_norm']")

def test_normalization_gradient(norm_layer, input_shape: Tuple[int, ...], 
                              eps: float = 1e-5) -> bool:
    """
    Test normalization gradient computation using finite differences.
    
    Args:
        norm_layer: Normalization layer instance
        input_shape: Shape of test input
        eps: Epsilon for finite differences
        
    Returns:
        True if gradients are approximately correct
    """
    print(f"Testing {norm_layer.__class__.__name__} gradients...")
    
    # Create random input
    np.random.seed(42)
    x = np.random.randn(*input_shape).astype(norm_layer.dtype)
    
    # Forward pass
    output = norm_layer.forward(x)
    
    # Create random upstream gradient
    grad_output = np.random.randn(*output.shape).astype(norm_layer.dtype)
    
    # Compute analytical gradients
    grad_input = norm_layer.backward(grad_output)
    
    # Compute numerical gradients using finite differences
    grad_input_numerical = np.zeros_like(x)
    
    # Test a subset of elements for efficiency
    test_indices = [(0, 0, 0), (0, 0, input_shape[-1]//2), (0, 0, input_shape[-1]-1)]
    if len(input_shape) > 3:
        test_indices = [(0, 0, 0, 0), (0, 0, 0, input_shape[-1]//2)]
    
    for idx in test_indices:
        if len(idx) <= len(input_shape):
            # Positive perturbation
            x_pos = x.copy()
            x_pos[idx] += eps
            output_pos = norm_layer.forward(x_pos)
            loss_pos = np.sum(grad_output * output_pos)
            
            # Negative perturbation
            x_neg = x.copy()
            x_neg[idx] -= eps
            output_neg = norm_layer.forward(x_neg)
            loss_neg = np.sum(grad_output * output_neg)
            
            # Finite difference gradient
            grad_numerical = (loss_pos - loss_neg) / (2 * eps)
            grad_analytical = grad_input[idx]
            
            # Check relative error
            rel_error = abs(grad_analytical - grad_numerical) / (abs(grad_numerical) + 1e-8)
            
            print(f"  Index {idx}: analytical={grad_analytical:.6f}, "
                  f"numerical={grad_numerical:.6f}, rel_error={rel_error:.6f}")
            
            if rel_error > 1e-3:  # Tolerance for numerical precision
                print(f"  WARNING: Large gradient error at {idx}")
                return False
    
    print("  Gradient test passed!")
    return True

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LAYER NORMALIZATION TESTING")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    seq_len = 20
    embed_dim = 128
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    
    print(f"Test input shape: {test_input.shape}")
    print_array_stats(test_input, "Input")
    
    # Test LayerNorm
    print(f"\n1. TESTING LAYER NORMALIZATION")
    print("-" * 40)
    
    layer_norm = LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
    ln_output = layer_norm.forward(test_input)
    
    print(f"LayerNorm output shape: {ln_output.shape}")
    print_array_stats(ln_output, "LayerNorm output")
    
    # Check normalization properties
    mean_after = np.mean(ln_output, axis=-1)
    std_after = np.std(ln_output, axis=-1)
    
    print(f"Mean after normalization (should be ~0): {np.mean(mean_after):.6f} ± {np.std(mean_after):.6f}")
    print(f"Std after normalization (should be ~1): {np.mean(std_after):.6f} ± {np.std(std_after):.6f}")
    
    # Test RMSNorm
    print(f"\n2. TESTING RMS NORMALIZATION")
    print("-" * 40)
    
    rms_norm = RMSNorm(embed_dim, eps=1e-8, elementwise_affine=True)
    rms_output = rms_norm.forward(test_input)
    
    print(f"RMSNorm output shape: {rms_output.shape}")
    print_array_stats(rms_output, "RMSNorm output")
    
    # Check RMS properties
    rms_after = np.sqrt(np.mean(rms_output**2, axis=-1))
    print(f"RMS after normalization (should be ~1): {np.mean(rms_after):.6f} ± {np.std(rms_after):.6f}")
    
    # Test gradient computation
    print(f"\n3. GRADIENT TESTING")
    print("-" * 40)
    
    # Test LayerNorm gradients
    success = test_normalization_gradient(layer_norm, test_input.shape)
    
    # Test RMSNorm gradients
    success &= test_normalization_gradient(rms_norm, test_input.shape)
    
    # Test backward pass
    print(f"\n4. BACKWARD PASS TESTING")
    print("-" * 40)
    
    # Create dummy gradients
    dummy_grad = np.random.randn(*ln_output.shape).astype(np.float32) * 0.01
    
    # LayerNorm backward
    layer_norm.zero_grad()
    ln_grad_input = layer_norm.backward(dummy_grad)
    
    print(f"LayerNorm gradient input shape: {ln_grad_input.shape}")
    print_array_stats(ln_grad_input, "LayerNorm grad input")
    
    ln_params = layer_norm.get_parameters()
    ln_grads = layer_norm.get_gradients()
    
    print(f"LayerNorm parameter gradients:")
    for name, grad in ln_grads.items():
        print(f"  {name}: {grad.shape}, mean={np.mean(grad):.6f}, std={np.std(grad):.6f}")
        check_finite(grad, f"LayerNorm {name} gradient")
    
    # RMSNorm backward
    rms_norm.zero_grad()
    rms_grad_input = rms_norm.backward(dummy_grad)
    
    print(f"RMSNorm gradient input shape: {rms_grad_input.shape}")
    print_array_stats(rms_grad_input, "RMSNorm grad input")
    
    rms_grads = rms_norm.get_gradients()
    
    print(f"RMSNorm parameter gradients:")
    for name, grad in rms_grads.items():
        print(f"  {name}: {grad.shape}, mean={np.mean(grad):.6f}, std={np.std(grad):.6f}")
        check_finite(grad, f"RMSNorm {name} gradient")
    
    # Test factory function
    print(f"\n5. FACTORY FUNCTION TESTING")
    print("-" * 40)
    
    # Create layers using factory
    ln_factory = create_norm_layer('layer_norm', embed_dim, eps=1e-5)
    rms_factory = create_norm_layer('rms_norm', embed_dim, eps=1e-8)
    
    print(f"Factory LayerNorm: {type(ln_factory).__name__}")
    print(f"Factory RMSNorm: {type(rms_factory).__name__}")
    
    # Test with different shapes
    print(f"\n6. DIFFERENT SHAPE TESTING")
    print("-" * 40)
    
    # Test 2D input (no batch dimension)
    test_2d = np.random.randn(seq_len, embed_dim).astype(np.float32)
    ln_2d = LayerNorm(embed_dim)
    output_2d = ln_2d.forward(test_2d)
    
    print(f"2D input shape: {test_2d.shape}")
    print(f"2D output shape: {output_2d.shape}")
    
    # Test without affine transformation
    print(f"\n7. NO AFFINE TRANSFORMATION TESTING")
    print("-" * 40)
    
    ln_no_affine = LayerNorm(embed_dim, elementwise_affine=False)
    output_no_affine = ln_no_affine.forward(test_input)
    
    print(f"No affine LayerNorm output shape: {output_no_affine.shape}")
    
    # Should have no parameters
    params_no_affine = ln_no_affine.get_parameters()
    print(f"Parameters (should be empty): {params_no_affine}")
    
    print("\n" + "=" * 60)
    if success:
        print("All layer normalization tests completed successfully!")
    else:
        print("Some tests failed - check gradient computation")
    print("Ready for integration with transformer blocks!")
    print("=" * 60)