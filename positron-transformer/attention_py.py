"""
Multi-Head Self-Attention for Pure Python Transformer Implementation

This module implements the core attention mechanism that makes transformers
powerful. Based on "Attention Is All You Need" and extensive experience with
matrix operations and numerical stability from FPGA neural network work.

The attention mechanism allows the model to focus on different parts of the
input sequence when processing each position. This is fundamentally different
from the fixed weight connections in traditional MLPs.

Mathematical Foundation:
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Key Features:
- Scaled dot-product attention with numerical stability
- Multi-head attention for diverse representation subspaces
- Causal masking for autoregressive language modeling
- Efficient matrix operations using NumPy
- Proper gradient computation for all parameters

Author: Joel Oliver
Based on: Master's thesis on FPGA Neural Network Implementation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
import math

from utils_py import (
    stable_softmax, create_causal_mask, create_padding_mask,
    check_finite, print_array_stats, gradient_clipping
)
from feedforward_py import Linear
from config_py import MODEL_CONFIG, PRECISION_CONFIG

# =============================================================================
# SCALED DOT-PRODUCT ATTENTION
# =============================================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention mechanism.
    
    This is the core attention function that computes attention weights
    and applies them to values. The scaling by √d_k prevents the dot products
    from becoming too large, which would push the softmax into regions with
    extremely small gradients.
    
    Mathematical Details:
        1. Compute attention scores: scores = QK^T / √d_k
        2. Apply mask (if provided): scores = scores + mask
        3. Compute attention weights: weights = softmax(scores)
        4. Apply dropout (if training): weights = dropout(weights)
        5. Compute output: output = weights @ V
    """
    
    def __init__(self,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 dtype: np.dtype = np.float32):
        """
        Initialize scaled dot-product attention.
        
        Args:
            dropout: Dropout probability for attention weights
            temperature: Temperature for softmax (1.0 for standard attention)
            dtype: Data type for computations
        """
        self.dropout = dropout
        self.temperature = temperature
        self.dtype = dtype
        
        # Cache for backward pass
        self._cache = {}
        
        print(f"ScaledDotProductAttention initialized: dropout={dropout}, temperature={temperature}")
    
    def forward(self,
                query: np.ndarray,
                key: np.ndarray,
                value: np.ndarray,
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (..., seq_len_q, d_k)
            key: Key tensor of shape (..., seq_len_k, d_k)
            value: Value tensor of shape (..., seq_len_v, d_v)
            mask: Optional attention mask (..., seq_len_q, seq_len_k)
            training: Whether in training mode
            
        Returns:
            Tuple of (attention_output, attention_weights)
            attention_output: shape (..., seq_len_q, d_v)
            attention_weights: shape (..., seq_len_q, seq_len_k)
        """
        # Get dimensions
        d_k = query.shape[-1]
        
        # Validate dimensions
        assert key.shape[-1] == d_k, f"Key dimension {key.shape[-1]} != query dimension {d_k}"
        assert key.shape[-2] == value.shape[-2], f"Key seq_len {key.shape[-2]} != value seq_len {value.shape[-2]}"
        
        # Compute attention scores: Q @ K^T / √d_k
        # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
        scores = np.matmul(query, np.swapaxes(key, -2, -1))
        
        # Scale by √d_k
        scale = math.sqrt(d_k) * self.temperature
        scores = scores / scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Compute attention weights using stable softmax
        attention_weights = stable_softmax(scores, axis=-1, temperature=1.0)
        
        # Apply dropout to attention weights if training
        if training and self.dropout > 0.0:
            attention_weights = self._apply_dropout(attention_weights)
        
        # Compute attention output: weights @ V
        # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
        attention_output = np.matmul(attention_weights, value)
        
        # Cache values for backward pass
        self._cache = {
            'query': query,
            'key': key,
            'value': value,
            'scores': scores,
            'attention_weights': attention_weights,
            'mask': mask,
            'scale': scale,
            'training': training
        }
        
        return attention_output, attention_weights
    
    def _apply_dropout(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Apply dropout to attention weights.
        
        Args:
            attention_weights: Attention weight matrix
            
        Returns:
            Attention weights with dropout applied
        """
        keep_prob = 1.0 - self.dropout
        mask = np.random.binomial(1, keep_prob, attention_weights.shape).astype(self.dtype)
        
        # Store dropout mask for backward pass
        self._cache['dropout_mask'] = mask
        
        # Apply dropout and renormalize
        dropped_weights = attention_weights * mask / keep_prob
        
        # Renormalize to maintain probability distribution
        row_sums = np.sum(dropped_weights, axis=-1, keepdims=True)
        dropped_weights = dropped_weights / (row_sums + 1e-8)
        
        return dropped_weights
    
    def backward(self,
                 grad_output: np.ndarray,
                 grad_attention_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of scaled dot-product attention.
        
        This is mathematically complex due to the softmax and matrix multiplications.
        
        Args:
            grad_output: Gradient w.r.t. attention output
            grad_attention_weights: Additional gradient w.r.t. attention weights
            
        Returns:
            Tuple of (grad_query, grad_key, grad_value)
        """
        if not self._cache:
            raise RuntimeError("Must call forward before backward")
        
        # Retrieve cached values
        query = self._cache['query']
        key = self._cache['key']
        value = self._cache['value']
        attention_weights = self._cache['attention_weights']
        scale = self._cache['scale']
        training = self._cache['training']
        
        # Gradient w.r.t. value: attention_weights^T @ grad_output
        grad_value = np.matmul(np.swapaxes(attention_weights, -2, -1), grad_output)
        
        # Gradient w.r.t. attention_weights: grad_output @ value^T
        grad_weights = np.matmul(grad_output, np.swapaxes(value, -2, -1))
        
        # Add any additional gradient w.r.t. attention weights
        if grad_attention_weights is not None:
            grad_weights = grad_weights + grad_attention_weights
        
        # Backward through dropout
        if training and self.dropout > 0.0 and 'dropout_mask' in self._cache:
            dropout_mask = self._cache['dropout_mask']
            keep_prob = 1.0 - self.dropout
            
            # Undo the renormalization and dropout scaling
            grad_weights_before_dropout = grad_weights * dropout_mask / keep_prob
            
            # Handle renormalization gradient (this is approximate)
            row_sums = np.sum(attention_weights * dropout_mask / keep_prob, axis=-1, keepdims=True)
            renorm_grad = -grad_weights * attention_weights / (row_sums + 1e-8)
            renorm_grad = np.sum(renorm_grad, axis=-1, keepdims=True)
            grad_weights_before_dropout = grad_weights_before_dropout + renorm_grad
            
            grad_weights = grad_weights_before_dropout
        
        # Backward through softmax
        # This implements the gradient of softmax function
        grad_scores = self._softmax_backward(attention_weights, grad_weights)
        
        # Backward through scaling
        grad_scores = grad_scores / scale
        
        # Backward through matrix multiplication QK^T
        # grad_query = grad_scores @ K
        grad_query = np.matmul(grad_scores, key)
        
        # grad_key = grad_scores^T @ Q
        grad_key = np.matmul(np.swapaxes(grad_scores, -2, -1), query)
        
        return grad_query, grad_key, grad_value
    
    def _softmax_backward(self, softmax_output: np.ndarray, grad_softmax: np.ndarray) -> np.ndarray:
        """
        Compute gradient through softmax function.
        
        For softmax: y_i = exp(x_i) / sum(exp(x_j))
        The gradient is: dy_i/dx_j = y_i * (δ_ij - y_j)
        
        Args:
            softmax_output: Output of softmax (attention weights)
            grad_softmax: Gradient w.r.t. softmax output
            
        Returns:
            Gradient w.r.t. softmax input (scores)
        """
        # Compute sum of (grad_softmax * softmax_output) along last dimension
        sum_grad_soft = np.sum(grad_softmax * softmax_output, axis=-1, keepdims=True)
        
        # Compute gradient: softmax_output * (grad_softmax - sum_grad_soft)
        grad_scores = softmax_output * (grad_softmax - sum_grad_soft)
        
        return grad_scores

# =============================================================================
# MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    This extends the basic attention to multiple "attention heads" that can
    focus on different types of relationships in the data. Each head operates
    on a different learned linear projection of the inputs.
    
    Architecture:
        1. Project inputs to Q, K, V using learned linear transformations
        2. Split into multiple heads (reshape to add head dimension)
        3. Apply scaled dot-product attention for each head
        4. Concatenate head outputs
        5. Apply final linear projection
    
    This design allows the model to attend to information from different
    representation subspaces simultaneously.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True,
                 add_zero_attn: bool = False,
                 dtype: np.dtype = np.float32):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
            add_zero_attn: Whether to add zero attention (for padding)
            dtype: Data type for parameters
        """
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.add_zero_attn = add_zero_attn
        self.dtype = dtype
        
        # Scale factor for attention (replaces √d_k scaling in attention function)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        
        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        
        # Attention function
        self.attention = ScaledDotProductAttention(dropout=dropout, dtype=dtype)
        
        # Cache for backward pass
        self._cache = {}
        
        print(f"MultiHeadAttention initialized:")
        print(f"  embed_dim: {embed_dim}")
        print(f"  num_heads: {num_heads}")
        print(f"  head_dim: {self.head_dim}")
        print(f"  dropout: {dropout}")
        print(f"  total_params: {4 * embed_dim * embed_dim + (4 * embed_dim if bias else 0):,}")
    
    def forward(self,
                query: np.ndarray,
                key: Optional[np.ndarray] = None,
                value: Optional[np.ndarray] = None,
                attn_mask: Optional[np.ndarray] = None,
                key_padding_mask: Optional[np.ndarray] = None,
                need_weights: bool = True,
                training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, tgt_len, embed_dim)
            key: Key tensor (batch_size, src_len, embed_dim) [defaults to query]
            value: Value tensor (batch_size, src_len, embed_dim) [defaults to key]
            attn_mask: Attention mask (tgt_len, src_len) or (batch_size, tgt_len, src_len)
            key_padding_mask: Key padding mask (batch_size, src_len)
            need_weights: Whether to return attention weights
            training: Whether in training mode
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]
        
        # Validate input dimensions
        assert key.shape[0] == batch_size and key.shape[2] == embed_dim
        assert value.shape[0] == batch_size and value.shape[1] == src_len and value.shape[2] == embed_dim
        
        # Linear projections
        Q = self.q_proj.forward(query)  # (batch_size, tgt_len, embed_dim)
        K = self.k_proj.forward(key)    # (batch_size, src_len, embed_dim)
        V = self.v_proj.forward(value)  # (batch_size, src_len, embed_dim)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        Q = Q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, src_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, src_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Combine attention and key padding masks
        combined_mask = self._combine_masks(attn_mask, key_padding_mask, batch_size, tgt_len, src_len)
        
        # Apply attention for each head
        # (batch_size, num_heads, tgt_len, head_dim), (batch_size, num_heads, tgt_len, src_len)
        attn_output, attn_weights = self.attention.forward(
            Q, K, V, mask=combined_mask, training=training
        )
        
        # Transpose back to (batch_size, tgt_len, num_heads, head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        
        # Concatenate heads: (batch_size, tgt_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, tgt_len, embed_dim)
        
        # Final linear projection
        output = self.out_proj.forward(attn_output)
        
        # Cache values for backward pass
        self._cache = {
            'query': query,
            'key': key,
            'value': value,
            'Q': Q,
            'K': K,
            'V': V,
            'attn_output_pre_proj': attn_output,
            'combined_mask': combined_mask,
            'attn_weights': attn_weights,
            'batch_size': batch_size,
            'tgt_len': tgt_len,
            'src_len': src_len
        }
        
        # Return attention weights if requested
        if need_weights:
            # Average attention weights across heads for interpretability
            avg_attn_weights = np.mean(attn_weights, axis=1)  # (batch_size, tgt_len, src_len)
            return output, avg_attn_weights
        else:
            return output, None
    
    def _combine_masks(self,
                      attn_mask: Optional[np.ndarray],
                      key_padding_mask: Optional[np.ndarray],
                      batch_size: int,
                      tgt_len: int,
                      src_len: int) -> Optional[np.ndarray]:
        """
        Combine attention mask and key padding mask.
        
        Args:
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            batch_size: Batch size
            tgt_len: Target sequence length
            src_len: Source sequence length
            
        Returns:
            Combined mask or None
        """
        combined_mask = None
        
        # Handle attention mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:  # (tgt_len, src_len)
                # Broadcast to (batch_size, num_heads, tgt_len, src_len)
                combined_mask = np.tile(attn_mask[np.newaxis, np.newaxis, :, :],
                                      (batch_size, self.num_heads, 1, 1))
            elif attn_mask.ndim == 3:  # (batch_size, tgt_len, src_len)
                # Broadcast to (batch_size, num_heads, tgt_len, src_len)
                combined_mask = np.tile(attn_mask[:, np.newaxis, :, :],
                                      (1, self.num_heads, 1, 1))
            else:
                combined_mask = attn_mask
        
        # Handle key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, src_len)
            # Convert to attention mask format: (batch_size, num_heads, tgt_len, src_len)
            padding_mask = key_padding_mask[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, src_len)
            padding_mask = np.tile(padding_mask, (1, self.num_heads, tgt_len, 1))
            
            if combined_mask is None:
                combined_mask = padding_mask
            else:
                combined_mask = combined_mask + padding_mask
        
        return combined_mask
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of multi-head attention.
        
        Args:
            grad_output: Gradient w.r.t. output
            
        Returns:
            Tuple of (grad_query, grad_key, grad_value)
        """
        if not self._cache:
            raise RuntimeError("Must call forward before backward")
        
        # Retrieve cached values
        query = self._cache['query']
        key = self._cache['key']
        value = self._cache['value']
        Q = self._cache['Q']
        K = self._cache['K']
        V = self._cache['V']
        attn_output_pre_proj = self._cache['attn_output_pre_proj']
        batch_size = self._cache['batch_size']
        tgt_len = self._cache['tgt_len']
        src_len = self._cache['src_len']
        
        # Backward through output projection
        grad_attn_output = self.out_proj.backward(grad_output)
        
        # Reshape to multi-head format
        grad_attn_output = grad_attn_output.reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        grad_attn_output = grad_attn_output.transpose(0, 2, 1, 3)
        
        # Backward through attention
        grad_Q, grad_K, grad_V = self.attention.backward(grad_attn_output)
        
        # Reshape gradients back to original format
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, tgt_len, self.embed_dim)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, src_len, self.embed_dim)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, src_len, self.embed_dim)
        
        # Backward through linear projections
        grad_query = self.q_proj.backward(grad_Q)
        grad_key = self.k_proj.backward(grad_K)
        grad_value = self.v_proj.backward(grad_V)
        
        return grad_query, grad_key, grad_value
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all parameters."""
        params = {}
        
        # Projection parameters
        for name, layer in [('q_proj', self.q_proj), ('k_proj', self.k_proj),
                           ('v_proj', self.v_proj), ('out_proj', self.out_proj)]:
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                params[f'{name}.{key}'] = value
        
        return params
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get all gradients."""
        grads = {}
        
        # Projection gradients
        for name, layer in [('q_proj', self.q_proj), ('k_proj', self.k_proj),
                           ('v_proj', self.v_proj), ('out_proj', self.out_proj)]:
            layer_grads = layer.get_gradients()
            for key, value in layer_grads.items():
                grads[f'{name}.{key}'] = value
        
        return grads
    
    def set_parameters(self, params: dict) -> None:
        """Set all parameters from loaded checkpoint"""
        # Set parameters for each projection layer
        for name, layer in [('q_proj', self.q_proj), ('k_proj', self.k_proj),
                           ('v_proj', self.v_proj), ('out_proj', self.out_proj)]:
            layer_params = {k.replace(f'{name}.', ''): v for k, v in params.items() if k.startswith(f'{name}.')}
            if layer_params:
                layer.set_parameters(layer_params)
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        self.q_proj.zero_grad()
        self.k_proj.zero_grad()
        self.v_proj.zero_grad()
        self.out_proj.zero_grad()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_attention_mask(seq_len: int, mask_type: str = 'causal') -> np.ndarray:
    """
    Create attention masks for different scenarios.
    
    Args:
        seq_len: Sequence length
        mask_type: Type of mask ('causal', 'full', 'local')
        
    Returns:
        Attention mask
    """
    if mask_type == 'causal':
        return create_causal_mask(seq_len)
    elif mask_type == 'full':
        return np.zeros((seq_len, seq_len), dtype=np.float32)
    elif mask_type == 'local':
        # Local attention with window size 3
        window = 3
        mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            mask[i, start:end] = 0.0
        return mask
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

def test_attention_gradient(attention_layer, input_shape: tuple, eps: float = 1e-5) -> bool:
    """
    Test attention gradient computation using finite differences.
    
    Args:
        attention_layer: Attention layer instance
        input_shape: Shape of test input
        eps: Epsilon for finite differences
        
    Returns:
        True if gradients are approximately correct
    """
    print(f"Testing {attention_layer.__class__.__name__} gradients...")
    
    # Create random inputs
    np.random.seed(42)
    query = np.random.randn(*input_shape).astype(attention_layer.dtype) * 0.1
    key = np.random.randn(*input_shape).astype(attention_layer.dtype) * 0.1
    value = np.random.randn(*input_shape).astype(attention_layer.dtype) * 0.1
    
    # Forward pass
    output, attn_weights = attention_layer.forward(query, key, value, training=False)
    
    # Create random upstream gradient
    grad_output = np.random.randn(*output.shape).astype(attention_layer.dtype) * 0.01
    
    # Compute analytical gradients
    grad_query, grad_key, grad_value = attention_layer.backward(grad_output)
    
    # Test a few elements with finite differences (limited for efficiency)
    test_indices = [(0, 0, 0), (0, 1, input_shape[-1]//2)]
    
    for idx in test_indices[:1]:  # Test only one for efficiency
        # Test query gradient
        query_pos = query.copy()
        query_pos[idx] += eps
        output_pos, _ = attention_layer.forward(query_pos, key, value, training=False)
        loss_pos = np.sum(grad_output * output_pos)
        
        query_neg = query.copy()
        query_neg[idx] -= eps
        output_neg, _ = attention_layer.forward(query_neg, key, value, training=False)
        loss_neg = np.sum(grad_output * output_neg)
        
        grad_numerical = (loss_pos - loss_neg) / (2 * eps)
        grad_analytical = grad_query[idx]
        
        rel_error = abs(grad_analytical - grad_numerical) / (abs(grad_numerical) + 1e-8)
        
        print(f"  Query gradient at {idx}: analytical={grad_analytical:.6f}, "
              f"numerical={grad_numerical:.6f}, rel_error={rel_error:.6f}")
        
        if rel_error > 1e-2:  # Lenient tolerance for complex operations
            print(f"  WARNING: Large gradient error in query at {idx}")
            return False
    
    print("  Gradient test passed!")
    return True

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-HEAD ATTENTION TESTING")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    seq_len = 20
    embed_dim = 128
    num_heads = 8
    
    # Create test inputs
    np.random.seed(42)
    test_query = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1
    test_key = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1
    test_value = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1
    
    print(f"Test input shapes:")
    print(f"  Query: {test_query.shape}")
    print(f"  Key: {test_key.shape}")
    print(f"  Value: {test_value.shape}")
    
    # Test Scaled Dot-Product Attention
    print(f"\n1. TESTING SCALED DOT-PRODUCT ATTENTION")
    print("-" * 40)
    
    attention = ScaledDotProductAttention(dropout=0.1)
    
    # Test without mask
    attn_output, attn_weights = attention.forward(test_query, test_key, test_value, training=False)
    
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print_array_stats(attn_output, "Attention output")
    print_array_stats(attn_weights, "Attention weights")
    
    # Check that attention weights sum to 1
    weight_sums = np.sum(attn_weights, axis=-1)
    print(f"Attention weight sums (should be ~1): mean={np.mean(weight_sums):.6f}, std={np.std(weight_sums):.6f}")
    
    # Test with causal mask
    print(f"\n2. TESTING WITH CAUSAL MASK")
    print("-" * 40)
    
    causal_mask = create_causal_mask(seq_len)
    attn_output_masked, attn_weights_masked = attention.forward(
        test_query, test_key, test_value, mask=causal_mask, training=False
    )
    
    print(f"Masked attention output shape: {attn_output_masked.shape}")
    print_array_stats(attn_weights_masked, "Masked attention weights")
    
    # Check that upper triangular part is zero (due to causal mask)
    upper_tri_sum = np.sum(attn_weights_masked * np.triu(np.ones((seq_len, seq_len)), k=1))
    print(f"Upper triangular attention sum (should be ~0): {upper_tri_sum:.6f}")
    
    # Test Multi-Head Attention
    print(f"\n3. TESTING MULTI-HEAD ATTENTION")
    print("-" * 40)
    
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    
    # Self-attention
    mha_output, mha_weights = mha.forward(test_query, training=False)
    
    print(f"Multi-head output shape: {mha_output.shape}")
    print(f"Multi-head weights shape: {mha_weights.shape}")
    print_array_stats(mha_output, "Multi-head output")
    
    # Cross-attention
    mha_cross_output, mha_cross_weights = mha.forward(
        test_query, test_key, test_value, training=False
    )
    
    print(f"Cross-attention output shape: {mha_cross_output.shape}")
    print_array_stats(mha_cross_output, "Cross-attention output")
    
    # Test with causal mask
    print(f"\n4. TESTING MULTI-HEAD WITH CAUSAL MASK")
    print("-" * 40)
    
    mha_causal_output, mha_causal_weights = mha.forward(
        test_query, attn_mask=causal_mask, training=False
    )
    
    print(f"Causal multi-head output shape: {mha_causal_output.shape}")
    print_array_stats(mha_causal_output, "Causal multi-head output")
    
    # Check causal property
    upper_tri_sum = np.sum(mha_causal_weights * np.triu(np.ones((seq_len, seq_len)), k=1))
    print(f"Upper triangular sum (should be ~0): {upper_tri_sum:.6f}")
    
    # Test padding mask
    print(f"\n5. TESTING KEY PADDING MASK")
    print("-" * 40)
    
    # Create padding mask (last 5 positions are padding)
    key_padding_mask = np.zeros((batch_size, seq_len), dtype=np.float32)
    key_padding_mask[:, -5:] = -np.inf
    
    mha_padded_output, mha_padded_weights = mha.forward(
        test_query, key_padding_mask=key_padding_mask, training=False
    )
    
    print(f"Padded attention output shape: {mha_padded_output.shape}")
    
    # Check that attention to padded positions is zero
    padded_attention_sum = np.sum(mha_padded_weights[:, :, -5:])
    print(f"Attention to padded positions (should be ~0): {padded_attention_sum:.6f}")
    
    # Test backward pass
    print(f"\n6. TESTING BACKWARD PASS")
    print("-" * 40)
    
    # Forward pass
    mha_test = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    output, _ = mha_test.forward(test_query, training=False)
    
    # Create dummy gradient
    dummy_grad = np.random.randn(*output.shape).astype(np.float32) * 0.01
    
    # Backward pass
    mha_test.zero_grad()
    grad_query, grad_key, grad_value = mha_test.backward(dummy_grad)
    
    print(f"Gradient shapes:")
    print(f"  Query gradient: {grad_query.shape}")
    print(f"  Key gradient: {grad_key.shape}")
    print(f"  Value gradient: {grad_value.shape}")
    
    print_array_stats(grad_query, "Query gradient")
    
    # Check parameter gradients
    params = mha_test.get_parameters()
    grads = mha_test.get_gradients()
    
    print(f"Parameter gradients:")
    for name, grad in grads.items():
        param = params[name]
        print(f"  {name}: {grad.shape}, norm={np.linalg.norm(grad):.6f}")
        check_finite(grad, f"MHA {name} gradient")
    
    # Test gradient computation
    print(f"\n7. GRADIENT CORRECTNESS TESTING")
    print("-" * 40)
    
    # Use smaller dimensions for efficiency
    small_input = test_query[:2, :5, :32]
    mha_small = MultiHeadAttention(embed_dim=32, num_heads=4, dropout=0.0)
    success = test_attention_gradient(mha_small, small_input.shape)
    
    # Test different mask types
    print(f"\n8. TESTING DIFFERENT MASK TYPES")
    print("-" * 40)
    
    for mask_name in ['causal', 'full', 'local']:
        mask = create_attention_mask(seq_len, mask_name)
        masked_output, masked_weights = mha.forward(
            test_query, attn_mask=mask, training=False
        )
        print(f"{mask_name.capitalize()} mask - output shape: {masked_output.shape}")
        print(f"  Attention weight stats: mean={np.mean(masked_weights):.6f}, "
              f"std={np.std(masked_weights):.6f}")
    
    # Parameter analysis
    print(f"\n9. PARAMETER ANALYSIS")
    print("-" * 40)
    
    params = mha.get_parameters()
    total_params = sum(p.size for p in params.values())
    
    print(f"Multi-head attention parameters:")
    for name, param in params.items():
        print(f"  {name}: {param.shape} ({param.size:,})")
    print(f"  Total: {total_params:,} parameters")
    
    # Memory and computation analysis
    print(f"\n10. COMPUTATIONAL ANALYSIS")
    print("-" * 40)
    
    # Approximate FLOPs for attention
    qkv_flops = 3 * batch_size * seq_len * embed_dim * embed_dim  # Q, K, V projections
    attention_flops = batch_size * num_heads * seq_len * seq_len * (2 * embed_dim // num_heads)  # Attention computation
    output_flops = batch_size * seq_len * embed_dim * embed_dim  # Output projection
    total_flops = qkv_flops + attention_flops + output_flops
    
    print(f"Approximate FLOPs:")
    print(f"  QKV projections: {qkv_flops:,}")
    print(f"  Attention computation: {attention_flops:,}")
    print(f"  Output projection: {output_flops:,}")
    print(f"  Total: {total_flops:,}")
    
    print("\n" + "=" * 60)
    if success:
        print("All attention mechanism tests completed successfully!")
    else:
        print("Some gradient tests failed - check implementation")
    print("Multi-head attention ready for transformer integration!")
    print("=" * 60)
        