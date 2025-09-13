"""
Feed-Forward Network for Pure Python Transformer Implementation

This module implements the feed-forward network (FFN) component of transformer
blocks. Based on extensive experience with MLP implementations from FPGA
neural network work, this provides a familiar foundation while adapting to
transformer-specific requirements.

The FFN in transformers typically follows this pattern:
    FFN(x) = activation(xW₁ + b₁)W₂ + b₂
    
Where the first layer expands the dimension (usually by 4x) and the second
layer projects back to the original dimension. This provides the model with
position-wise transformation capabilities.

Key Features:
- Multiple activation function support (ReLU, GELU, Swish)
- Configurable hidden dimension scaling
- Dropout for regularization
- Efficient matrix operations using NumPy
- Proper gradient computation for backpropagation

Author: Joel Oliver
Based on: Master's thesis on FPGA Neural Network Implementation
"""

import numpy as np
from typing import Dict, Optional, Callable, Union

from utils_py import (
    get_activation_function, get_activation_derivative,
    xavier_normal_init, normal_init, zeros_init,
    check_finite, print_array_stats, gradient_clipping
)
from config_py import MODEL_CONFIG, PRECISION_CONFIG

# =============================================================================
# LINEAR LAYER
# =============================================================================

class Linear:
    """
    Linear transformation layer: y = xW + b
    
    This is the fundamental building block for feed-forward networks,
    similar to the linear components in MLP implementations from FPGA work.
    Handles weight initialization, forward/backward propagation, and gradient
    accumulation.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 init_type: str = 'xavier_normal',
                 init_std: Optional[float] = None,
                 dtype: np.dtype = np.float32):
        """
        Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            init_type: Weight initialization type ('xavier_normal', 'normal', 'xavier_uniform')
            init_std: Standard deviation for normal initialization
            dtype: Data type for parameters
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.dtype = dtype
        
        # Initialize weights
        self.weight = self._initialize_weights(init_type, init_std)
        
        # Initialize bias
        if self.use_bias:
            self.bias = zeros_init((out_features,), dtype=dtype)
            self.bias_grad = np.zeros_like(self.bias)
        else:
            self.bias = None
            self.bias_grad = None
        
        # Weight gradients
        self.weight_grad = np.zeros_like(self.weight)
        
        # Cache for backward pass
        self._input_cache = None
        
        print(f"Linear layer initialized: {in_features} → {out_features}, "
              f"bias={bias}, params={self.weight.size + (out_features if bias else 0):,}")
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving"""
        params = {'weight': self.weight}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from loaded checkpoint"""
        self.weight = params['weight']
        self.weight_grad = np.zeros_like(self.weight)
        if self.use_bias and 'bias' in params:
            self.bias = params['bias']
            self.bias_grad = np.zeros_like(self.bias)
    
    def _initialize_weights(self, init_type: str, init_std: Optional[float]) -> np.ndarray:
        """
        Initialize weights based on initialization type.
        
        Args:
            init_type: Type of initialization
            init_std: Standard deviation (for normal init)
            
        Returns:
            Initialized weight matrix
        """
        shape = (self.in_features, self.out_features)
        
        if init_type == 'xavier_normal':
            return xavier_normal_init(shape, gain=1.0, dtype=self.dtype)
        elif init_type == 'normal':
            std = init_std if init_std is not None else 0.02
            return normal_init(shape, std=std, dtype=self.dtype)
        elif init_type == 'xavier_uniform':
            from utils_py import xavier_uniform_init
            return xavier_uniform_init(shape, gain=1.0, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = xW + b
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Cache input for backward pass
        self._input_cache = x
        
        # Matrix multiplication: (..., in_features) @ (in_features, out_features)
        output = np.dot(x, self.weight)
        
        # Add bias if present
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients.
        
        Args:
            grad_output: Gradient from upstream layers
            
        Returns:
            Gradient with respect to input
        """
        if self._input_cache is None:
            raise RuntimeError("Must call forward before backward")
        
        x = self._input_cache
        
        # Gradient w.r.t. input: grad_output @ W.T
        grad_input = np.dot(grad_output, self.weight.T)
        
        # Gradient w.r.t. weight: x.T @ grad_output
        # Handle batch dimension properly
        if x.ndim == 2:  # (batch_size, in_features)
            self.weight_grad += np.dot(x.T, grad_output)
        elif x.ndim == 3:  # (batch_size, seq_len, in_features)
            # Reshape to 2D for matrix multiplication
            batch_size, seq_len, in_features = x.shape
            x_reshaped = x.reshape(-1, in_features)  # (batch_size * seq_len, in_features)
            grad_reshaped = grad_output.reshape(-1, self.out_features)  # (batch_size * seq_len, out_features)
            self.weight_grad += np.dot(x_reshaped.T, grad_reshaped)
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}")
        
        # Gradient w.r.t. bias: sum over batch and sequence dimensions
        if self.use_bias:
            if grad_output.ndim == 2:  # (batch_size, out_features)
                self.bias_grad += np.sum(grad_output, axis=0)
            elif grad_output.ndim == 3:  # (batch_size, seq_len, out_features)
                self.bias_grad += np.sum(grad_output, axis=(0, 1))
            else:
                raise ValueError(f"Unsupported gradient dimension: {grad_output.ndim}")
        
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters dictionary."""
        params = {'weight': self.weight}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients dictionary."""
        grads = {'weight': self.weight_grad}
        if self.use_bias:
            grads['bias'] = self.bias_grad
        return grads
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.weight_grad.fill(0.0)
        if self.use_bias:
            self.bias_grad.fill(0.0)

# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward:
    """
    Transformer Feed-Forward Network (FFN).
    
    Implements the position-wise feed-forward network used in transformer blocks.
    This is essentially a two-layer MLP with an activation function in between,
    similar to the neural network architectures from FPGA implementation experience.
    
    Architecture:
        Input → Linear1 → Activation → Dropout → Linear2 → Output
        
    The hidden dimension is typically 4x the input dimension in standard transformers.
    """
    
    def __init__(self,
                 embed_dim: int,
                 ff_dim: Optional[int] = None,
                 activation: str = 'gelu',
                 dropout: float = 0.1,
                 bias: bool = True,
                 init_type: str = 'xavier_normal',
                 dtype: np.dtype = np.float32):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Input/output embedding dimension
            ff_dim: Hidden layer dimension (defaults to 4 * embed_dim)
            activation: Activation function name
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            init_type: Weight initialization type
            dtype: Data type for parameters
        """
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim
        self.activation_name = activation
        self.dropout = dropout
        self.dtype = dtype
        
        # Get activation function and its derivative
        self.activation_fn = get_activation_function(activation)
        self.activation_derivative = get_activation_derivative(activation)
        
        # Linear layers
        self.linear1 = Linear(
            in_features=embed_dim,
            out_features=self.ff_dim,
            bias=bias,
            init_type=init_type,
            dtype=dtype
        )
        
        self.linear2 = Linear(
            in_features=self.ff_dim,
            out_features=embed_dim,
            bias=bias,
            init_type=init_type,
            dtype=dtype
        )
        
        # Cache for backward pass
        self._cache = {}
        
        total_params = (embed_dim * self.ff_dim + self.ff_dim * embed_dim +
                       (self.ff_dim + embed_dim if bias else 0))
        
        print(f"FeedForward initialized: {embed_dim} → {self.ff_dim} → {embed_dim}")
        print(f"  Activation: {activation}")
        print(f"  Dropout: {dropout}")
        print(f"  Parameters: {total_params:,}")
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (..., embed_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Output tensor of same shape as input
        """
        # First linear transformation
        hidden = self.linear1.forward(x)
        
        # Apply activation function
        activated = self.activation_fn(hidden)
        
        # Apply dropout in training mode
        if training and self.dropout > 0.0:
            activated_dropout = self._apply_dropout(activated)
        else:
            activated_dropout = activated
        
        # Second linear transformation
        output = self.linear2.forward(activated_dropout)
        
        # Cache intermediate values for backward pass
        self._cache = {
            'hidden': hidden,
            'activated': activated,
            'activated_dropout': activated_dropout,
            'training': training
        }
        
        return output
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout to input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with dropout applied
        """
        keep_prob = 1.0 - self.dropout
        mask = np.random.binomial(1, keep_prob, x.shape).astype(self.dtype)
        
        # Store mask for backward pass
        self._cache['dropout_mask'] = mask
        
        # Apply dropout and scale
        return x * mask / keep_prob
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through feed-forward network.
        
        Args:
            grad_output: Gradient from upstream layers
            
        Returns:
            Gradient with respect to input
        """
        if not self._cache:
            raise RuntimeError("Must call forward before backward")
        
        # Retrieve cached values
        hidden = self._cache['hidden']
        activated = self._cache['activated']
        activated_dropout = self._cache['activated_dropout']
        training = self._cache['training']
        
        # Backward through second linear layer
        grad_activated_dropout = self.linear2.backward(grad_output)
        
        # Backward through dropout
        if training and self.dropout > 0.0 and 'dropout_mask' in self._cache:
            dropout_mask = self._cache['dropout_mask']
            keep_prob = 1.0 - self.dropout
            grad_activated = grad_activated_dropout * dropout_mask / keep_prob
        else:
            grad_activated = grad_activated_dropout
        
        # Backward through activation function
        grad_hidden = grad_activated * self.activation_derivative(hidden)
        
        # Backward through first linear layer
        grad_input = self.linear1.backward(grad_hidden)
        
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all parameters."""
        params = {}
        
        # Linear1 parameters
        linear1_params = self.linear1.get_parameters()
        for key, value in linear1_params.items():
            params[f'linear1.{key}'] = value
        
        # Linear2 parameters
        linear2_params = self.linear2.get_parameters()
        for key, value in linear2_params.items():
            params[f'linear2.{key}'] = value
        
        return params
    
    def set_parameters(self, params: dict) -> None:
        """Set all parameters from loaded checkpoint"""
        # Extract linear1 parameters
        linear1_params = {k.replace('linear1.', ''): v for k, v in params.items() if k.startswith('linear1.')}
        if linear1_params:
            self.linear1.set_parameters(linear1_params)
        
        # Extract linear2 parameters
        linear2_params = {k.replace('linear2.', ''): v for k, v in params.items() if k.startswith('linear2.')}
        if linear2_params:
            self.linear2.set_parameters(linear2_params)
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get all gradients."""
        grads = {}
        
        # Linear1 gradients
        linear1_grads = self.linear1.get_gradients()
        for key, value in linear1_grads.items():
            grads[f'linear1.{key}'] = value
        
        # Linear2 gradients
        linear2_grads = self.linear2.get_gradients()
        for key, value in linear2_grads.items():
            grads[f'linear2.{key}'] = value
        
        return grads
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        self.linear1.zero_grad()
        self.linear2.zero_grad()

# =============================================================================
# SPECIALIZED FEED-FORWARD VARIANTS
# =============================================================================

class GLUFeedForward:
    """
    Gated Linear Unit (GLU) Feed-Forward Network.
    
    This variant uses gating mechanism which can be more expressive than
    standard FFN. The architecture is:
        FFN(x) = (xW₁ + b₁) ⊙ σ(xW₂ + b₂)
        
    Where ⊙ is element-wise multiplication and σ is a gating function (usually sigmoid).
    """
    
    def __init__(self,
                 embed_dim: int,
                 ff_dim: Optional[int] = None,
                 gate_activation: str = 'sigmoid',
                 dropout: float = 0.1,
                 bias: bool = True,
                 dtype: np.dtype = np.float32):
        """
        Initialize GLU feed-forward network.
        
        Args:
            embed_dim: Input/output embedding dimension
            ff_dim: Hidden dimension (defaults to 4 * embed_dim)
            gate_activation: Gating activation function
            dropout: Dropout probability
            bias: Whether to use bias
            dtype: Data type
        """
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim
        self.dropout = dropout
        self.dtype = dtype
        
        # Get gate activation function
        self.gate_fn = get_activation_function(gate_activation)
        self.gate_derivative = get_activation_derivative(gate_activation)
        
        # Linear layers for value and gate
        self.value_proj = Linear(embed_dim, self.ff_dim, bias=bias, dtype=dtype)
        self.gate_proj = Linear(embed_dim, self.ff_dim, bias=bias, dtype=dtype)
        self.output_proj = Linear(self.ff_dim, embed_dim, bias=bias, dtype=dtype)
        
        # Cache for backward pass
        self._cache = {}
        
        print(f"GLU FeedForward initialized: {embed_dim} → {self.ff_dim} → {embed_dim}")
        print(f"  Gate activation: {gate_activation}")
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through GLU FFN."""
        # Compute value and gate
        value = self.value_proj.forward(x)
        gate_logits = self.gate_proj.forward(x)
        gate = self.gate_fn(gate_logits)
        
        # Apply gating
        gated = value * gate
        
        # Apply dropout
        if training and self.dropout > 0.0:
            gated = self._apply_dropout(gated)
        
        # Output projection
        output = self.output_proj.forward(gated)
        
        # Cache for backward pass
        self._cache = {
            'value': value,
            'gate_logits': gate_logits,
            'gate': gate,
            'gated': gated,
            'training': training
        }
        
        return output
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout with mask caching."""
        keep_prob = 1.0 - self.dropout
        mask = np.random.binomial(1, keep_prob, x.shape).astype(self.dtype)
        self._cache['dropout_mask'] = mask
        return x * mask / keep_prob
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through GLU FFN."""
        if not self._cache:
            raise RuntimeError("Must call forward before backward")
        
        value = self._cache['value']
        gate_logits = self._cache['gate_logits']
        gate = self._cache['gate']
        gated = self._cache['gated']
        training = self._cache['training']
        
        # Backward through output projection
        grad_gated = self.output_proj.backward(grad_output)
        
        # Backward through dropout
        if training and self.dropout > 0.0 and 'dropout_mask' in self._cache:
            dropout_mask = self._cache['dropout_mask']
            keep_prob = 1.0 - self.dropout
            grad_gated = grad_gated * dropout_mask / keep_prob
        
        # Backward through gating
        grad_value = grad_gated * gate
        grad_gate = grad_gated * value
        
        # Backward through gate activation
        grad_gate_logits = grad_gate * self.gate_derivative(gate_logits)
        
        # Backward through projections
        grad_x_value = self.value_proj.backward(grad_value)
        grad_x_gate = self.gate_proj.backward(grad_gate_logits)
        
        # Combine gradients
        grad_input = grad_x_value + grad_x_gate
        
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all parameters."""
        params = {}
        
        for name, layer in [('value_proj', self.value_proj),
                           ('gate_proj', self.gate_proj),
                           ('output_proj', self.output_proj)]:
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                params[f'{name}.{key}'] = value
        
        return params
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get all gradients."""
        grads = {}
        
        for name, layer in [('value_proj', self.value_proj),
                           ('gate_proj', self.gate_proj),
                           ('output_proj', self.output_proj)]:
            layer_grads = layer.get_gradients()
            for key, value in layer_grads.items():
                grads[f'{name}.{key}'] = value
        
        return grads
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        self.value_proj.zero_grad()
        self.gate_proj.zero_grad()
        self.output_proj.zero_grad()

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_feedforward(ff_type: str = 'standard',
                      embed_dim: int = 512,
                      ff_dim: Optional[int] = None,
                      **kwargs) -> Union[FeedForward, GLUFeedForward]:
    """
    Factory function to create feed-forward networks.
    
    Args:
        ff_type: Type of FFN ('standard' or 'glu')
        embed_dim: Embedding dimension
        ff_dim: Feed-forward dimension
        **kwargs: Additional arguments
        
    Returns:
        Feed-forward network instance
    """
    if ff_type == 'standard':
        return FeedForward(embed_dim=embed_dim, ff_dim=ff_dim, **kwargs)
    elif ff_type == 'glu':
        return GLUFeedForward(embed_dim=embed_dim, ff_dim=ff_dim, **kwargs)
    else:
        raise ValueError(f"Unknown ff_type: {ff_type}. Supported: ['standard', 'glu']")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def test_feedforward_gradient(ff_layer, input_shape: tuple, eps: float = 1e-5) -> bool:
    """
    Test feed-forward gradient computation using finite differences.
    
    Args:
        ff_layer: Feed-forward layer instance
        input_shape: Shape of test input
        eps: Epsilon for finite differences
        
    Returns:
        True if gradients are approximately correct
    """
    print(f"Testing {ff_layer.__class__.__name__} gradients...")
    
    # Create random input
    np.random.seed(42)
    x = np.random.randn(*input_shape).astype(ff_layer.dtype) * 0.1
    
    # Forward pass
    output = ff_layer.forward(x, training=False)  # Disable dropout for testing
    
    # Create random upstream gradient
    grad_output = np.random.randn(*output.shape).astype(ff_layer.dtype) * 0.01
    
    # Compute analytical gradients
    grad_input = ff_layer.backward(grad_output)
    
    # Test a few input elements with finite differences
    test_indices = [(0, 0, 0), (0, 1, input_shape[-1]//2), (0, -1, -1)]
    
    for idx in test_indices[:2]:  # Test fewer for efficiency
        if len(idx) <= len(input_shape):
            # Positive perturbation
            x_pos = x.copy()
            x_pos[idx] += eps
            output_pos = ff_layer.forward(x_pos, training=False)
            loss_pos = np.sum(grad_output * output_pos)
            
            # Negative perturbation
            x_neg = x.copy()
            x_neg[idx] -= eps
            output_neg = ff_layer.forward(x_neg, training=False)
            loss_neg = np.sum(grad_output * output_neg)
            
            # Finite difference gradient
            grad_numerical = (loss_pos - loss_neg) / (2 * eps)
            grad_analytical = grad_input[idx]
            
            # Check relative error
            rel_error = abs(grad_analytical - grad_numerical) / (abs(grad_numerical) + 1e-8)
            
            print(f"  Index {idx}: analytical={grad_analytical:.6f}, "
                  f"numerical={grad_numerical:.6f}, rel_error={rel_error:.6f}")
            
            if rel_error > 1e-2:  # More lenient for complex networks
                print(f"  WARNING: Large gradient error at {idx}")
                return False
    
    print("  Gradient test passed!")
    return True

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEED-FORWARD NETWORK TESTING")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    seq_len = 20
    embed_dim = 128
    ff_dim = 512
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1
    
    print(f"Test input shape: {test_input.shape}")
    print_array_stats(test_input, "Input")
    
    # Test Linear layer
    print(f"\n1. TESTING LINEAR LAYER")
    print("-" * 40)
    
    linear = Linear(embed_dim, ff_dim, bias=True)
    linear_output = linear.forward(test_input)
    
    print(f"Linear output shape: {linear_output.shape}")
    print_array_stats(linear_output, "Linear output")
    
    # Test different activations
    activations = ['relu', 'gelu', 'swish']
    
    for i, activation in enumerate(activations):
        print(f"\n{i+2}. TESTING FEEDFORWARD WITH {activation.upper()}")
        print("-" * 40)
        
        ff = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            activation=activation,
            dropout=0.1,
            bias=True
        )
        
        # Forward pass (training mode)
        ff_output_train = ff.forward(test_input, training=True)
        print(f"FFN output (training) shape: {ff_output_train.shape}")
        print_array_stats(ff_output_train, f"FFN output ({activation}, training)")
        
        # Forward pass (eval mode)
        ff_output_eval = ff.forward(test_input, training=False)
        print(f"FFN output (eval) shape: {ff_output_eval.shape}")
        print_array_stats(ff_output_eval, f"FFN output ({activation}, eval)")
        
        # Check that outputs are different due to dropout
        if not np.allclose(ff_output_train, ff_output_eval):
            print("✓ Dropout working correctly (train ≠ eval)")
        else:
            print("⚠ Dropout may not be working (train = eval)")
    
    # Test GLU Feed-Forward
    print(f"\n5. TESTING GLU FEED-FORWARD")
    print("-" * 40)
    
    glu_ff = GLUFeedForward(
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        gate_activation='sigmoid',
        dropout=0.1
    )
    
    glu_output = glu_ff.forward(test_input, training=False)
    print(f"GLU FFN output shape: {glu_output.shape}")
    print_array_stats(glu_output, "GLU FFN output")
    
    # Test backward pass
    print(f"\n6. TESTING BACKWARD PASS")
    print("-" * 40)
    
    # Create FFN for gradient testing
    ff_test = FeedForward(embed_dim=embed_dim, ff_dim=ff_dim, activation='gelu', dropout=0.0)
    
    # Forward pass
    output = ff_test.forward(test_input, training=False)
    
    # Create dummy gradient
    dummy_grad = np.random.randn(*output.shape).astype(np.float32) * 0.01
    
    # Backward pass
    ff_test.zero_grad()
    grad_input = ff_test.backward(dummy_grad)
    
    print(f"Gradient input shape: {grad_input.shape}")
    print_array_stats(grad_input, "Input gradient")
    
    # Check parameter gradients
    params = ff_test.get_parameters()
    grads = ff_test.get_gradients()
    
    print(f"Parameter gradients:")
    for name, grad in grads.items():
        param = params[name]
        print(f"  {name}: {grad.shape}, mean={np.mean(grad):.6f}, "
              f"std={np.std(grad):.6f}, norm={np.linalg.norm(grad):.6f}")
        check_finite(grad, f"FFN {name} gradient")
    
    # Test gradient computation
    print(f"\n7. GRADIENT CORRECTNESS TESTING")
    print("-" * 40)
    
    # Test standard FFN
    small_input = test_input[:2, :5, :32]  # Smaller for efficiency
    ff_small = FeedForward(embed_dim=32, ff_dim=64, activation='relu', dropout=0.0)
    success = test_feedforward_gradient(ff_small, small_input.shape)
    
    # Test GLU FFN
    glu_small = GLUFeedForward(embed_dim=32, ff_dim=64, dropout=0.0)
    success &= test_feedforward_gradient(glu_small, small_input.shape)
    
    # Test factory function
    print(f"\n8. FACTORY FUNCTION TESTING")
    print("-" * 40)
    
    # Create using factory
    ff_factory = create_feedforward('standard', embed_dim=embed_dim, ff_dim=ff_dim, activation='gelu')
    glu_factory = create_feedforward('glu', embed_dim=embed_dim, ff_dim=ff_dim)
    
    print(f"Factory standard FFN: {type(ff_factory).__name__}")
    print(f"Factory GLU FFN: {type(glu_factory).__name__}")
    
    # Test outputs
    factory_output = ff_factory.forward(test_input, training=False)
    glu_factory_output = glu_factory.forward(test_input, training=False)
    
    print(f"Factory FFN output shape: {factory_output.shape}")
    print(f"Factory GLU output shape: {glu_factory_output.shape}")
    
    # Parameter count comparison
    print(f"\n9. PARAMETER ANALYSIS")
    print("-" * 40)
    
    ff_params = ff_test.get_parameters()
    glu_params = glu_ff.get_parameters()
    
    ff_total = sum(p.size for p in ff_params.values())
    glu_total = sum(p.size for p in glu_params.values())
    
    print(f"Standard FFN parameters:")
    for name, param in ff_params.items():
        print(f"  {name}: {param.shape} ({param.size:,})")
    print(f"  Total: {ff_total:,}")
    
    print(f"\nGLU FFN parameters:")
    for name, param in glu_params.items():
        print(f"  {name}: {param.shape} ({param.size:,})")
    print(f"  Total: {glu_total:,}")
    
    print(f"\nParameter ratio (GLU/Standard): {glu_total/ff_total:.2f}x")
    
    print("\n" + "=" * 60)
    if success:
        print("All feed-forward network tests completed successfully!")
    else:
        print("Some gradient tests failed - check implementation")
    print("Feed-forward networks ready for transformer integration!")
    print("=" * 60)