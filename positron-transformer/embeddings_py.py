"""
Token and Positional Embeddings for Pure Python Transformer Implementation

This module implements token embeddings and positional encodings for the
transformer model. Based on extensive experience with neural network
architectures and mathematical precision from FPGA implementation work.

Key Features:
- Learnable token embeddings with proper initialization
- Sinusoidal positional encoding (from "Attention Is All You Need")
- Learnable positional embeddings (alternative approach)
- Efficient batch processing with NumPy
- Numerical stability considerations

Mathematical Foundation:
- Token embeddings: Learnable lookup table mapping token_id -> embedding_vector
- Positional encoding: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
                      PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

Author: Joel Oliver
Based on: Master's thesis on FPGA Neural Network Implementation
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
import math

from utils_py import (
    normal_init, xavier_normal_init, zeros_init, 
    check_finite, print_array_stats
)
from config_py import MODEL_CONFIG, PRECISION_CONFIG

# =============================================================================
# TOKEN EMBEDDINGS
# =============================================================================

class TokenEmbedding:
    """
    Token embedding layer that converts token IDs to dense vectors.
    
    This layer maintains a learnable lookup table that maps each token ID
    to a dense embedding vector. Similar to the MLP layers from FPGA work,
    but specialized for embedding lookups.
    
    The embedding matrix is of shape (vocab_size, embed_dim) where each
    row corresponds to the embedding vector for a specific token.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 padding_idx: Optional[int] = None,
                 init_std: float = 0.02,
                 dtype: np.dtype = np.float32):
        """
        Initialize token embedding layer.
        
        Args:
            vocab_size: Size of vocabulary (number of possible tokens)
            embed_dim: Dimension of embedding vectors
            padding_idx: Index of padding token (embedding will be zero)
            init_std: Standard deviation for weight initialization
            dtype: Data type for embeddings
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.dtype = dtype
        
        # Initialize embedding matrix
        # Shape: (vocab_size, embed_dim)
        self.weight = self._initialize_weights()
        
        # Zero out padding embedding if specified
        if self.padding_idx is not None:
            self.weight[self.padding_idx, :] = 0.0
        
        # Store gradients for backpropagation
        self.weight_grad = np.zeros_like(self.weight)
        
        print(f"TokenEmbedding initialized: vocab_size={vocab_size}, "
              f"embed_dim={embed_dim}, params={self.weight.size:,}")
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving"""
        return {'weight': self.weight}
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from loaded checkpoint"""
        self.weight = params['weight']
        self.weight_grad = np.zeros_like(self.weight)
    
    def _initialize_weights(self) -> np.ndarray:
        """
        Initialize embedding weights using normal distribution.
        
        Based on experience with weight initialization from neural network work.
        Using normal initialization similar to word2vec and modern transformers.
        
        Returns:
            Initialized weight matrix
        """
        # Use normal initialization (common for embeddings)
        weights = normal_init(
            shape=(self.vocab_size, self.embed_dim),
            mean=0.0,
            std=self.init_std,
            dtype=self.dtype
        )
        
        return weights
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass: lookup embeddings for token IDs.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len) or (seq_len,)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        # Handle 1D input (single sequence)
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = token_ids.shape
        
        # Validate token IDs
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError(f"Token IDs must be in range [0, {self.vocab_size})")
        
        # Lookup embeddings
        # This is equivalent to: embeddings = self.weight[token_ids]
        # But we make it explicit for clarity
        embeddings = np.zeros((batch_size, seq_len, self.embed_dim), dtype=self.dtype)
        
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = int(token_ids[i, j])
                embeddings[i, j, :] = self.weight[token_id, :]
        
        # Remove batch dimension if input was 1D
        if squeeze_output:
            embeddings = embeddings.squeeze(0)
        
        return embeddings
    
    def backward(self, token_ids: np.ndarray, grad_output: np.ndarray) -> None:
        """
        Backward pass: compute gradients with respect to embedding weights.
        
        Args:
            token_ids: Token IDs used in forward pass
            grad_output: Gradient from upstream layers
        """
        # Handle 1D input
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
            if grad_output.ndim == 2:
                grad_output = grad_output[np.newaxis, :, :]
        
        batch_size, seq_len = token_ids.shape
        
        # Accumulate gradients for each token
        # Since multiple positions might use the same token, we accumulate
        self.weight_grad.fill(0.0)
        
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = int(token_ids[i, j])
                self.weight_grad[token_id, :] += grad_output[i, j, :]
        
        # Zero out gradients for padding token
        if self.padding_idx is not None:
            self.weight_grad[self.padding_idx, :] = 0.0
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters dictionary."""
        return {'weight': self.weight}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients dictionary."""
        return {'weight': self.weight_grad}

# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding:
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    
    This implementation uses the sinusoidal functions to create positional
    encodings that the model can use to understand sequence order. Unlike
    learnable positional embeddings, these are fixed mathematical functions.
    
    The key advantage is that this encoding can handle sequences longer
    than those seen during training.
    
    Mathematical Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where pos is the position and i is the dimension index.
    """
    
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int = 5000,
                 dtype: np.dtype = np.float32):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension (must be even for proper sin/cos pairing)
            max_seq_len: Maximum sequence length to precompute
            dtype: Data type for encodings
        """
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        
        # Precompute positional encodings
        self.pe = self._create_positional_encoding()
        
        print(f"PositionalEncoding initialized: embed_dim={embed_dim}, "
              f"max_seq_len={max_seq_len}")
    
    def _create_positional_encoding(self) -> np.ndarray:
        """
        Create sinusoidal positional encoding matrix.
        
        Returns:
            Positional encoding matrix of shape (max_seq_len, embed_dim)
        """
        # Initialize position encoding matrix
        pe = np.zeros((self.max_seq_len, self.embed_dim), dtype=self.dtype)
        
        # Create position indices
        position = np.arange(0, self.max_seq_len, dtype=self.dtype).reshape(-1, 1)
        
        # Create dimension indices (0, 2, 4, ..., embed_dim-2)
        div_term = np.exp(np.arange(0, self.embed_dim, 2, dtype=self.dtype) * 
                         -(math.log(10000.0) / self.embed_dim))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        if self.embed_dim % 2 == 1:
            # Handle odd embedding dimensions
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, seq_len: int, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Get positional encoding for sequences.
        
        Args:
            seq_len: Length of sequence
            batch_size: Batch size (if None, returns 2D array)
            
        Returns:
            Positional encodings of shape (seq_len, embed_dim) or 
            (batch_size, seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Get positional encodings for required length
        pe = self.pe[:seq_len, :]
        
        # Add batch dimension if requested
        if batch_size is not None:
            pe = np.tile(pe[np.newaxis, :, :], (batch_size, 1, 1))
        
        return pe
    
    def __call__(self, seq_len: int, batch_size: Optional[int] = None) -> np.ndarray:
        """Callable interface."""
        return self.forward(seq_len, batch_size)

# =============================================================================
# LEARNABLE POSITIONAL EMBEDDINGS
# =============================================================================

class LearnablePositionalEmbedding:
    """
    Learnable positional embeddings as an alternative to sinusoidal encoding.
    
    This approach learns position-specific embeddings during training.
    It's similar to token embeddings but for positions instead of tokens.
    
    Advantages:
    - Can learn task-specific positional patterns
    - Often performs slightly better than sinusoidal in practice
    
    Disadvantages:
    - Cannot handle sequences longer than max_seq_len seen during training
    - More parameters to learn
    """
    
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 init_std: float = 0.02,
                 dtype: np.dtype = np.float32):
        """
        Initialize learnable positional embeddings.
        
        Args:
            embed_dim: Embedding dimension
            max_seq_len: Maximum sequence length
            init_std: Standard deviation for initialization
            dtype: Data type for embeddings
        """
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.init_std = init_std
        self.dtype = dtype
        
        # Initialize positional embedding matrix
        # Shape: (max_seq_len, embed_dim)
        self.weight = normal_init(
            shape=(max_seq_len, embed_dim),
            std=init_std,
            dtype=dtype
        )
        
        # Gradient storage
        self.weight_grad = np.zeros_like(self.weight)
        
        print(f"LearnablePositionalEmbedding initialized: embed_dim={embed_dim}, "
              f"max_seq_len={max_seq_len}, params={self.weight.size:,}")
    
    def forward(self, seq_len: int, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Forward pass: get positional embeddings.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size (if None, returns 2D array)
            
        Returns:
            Positional embeddings of shape (seq_len, embed_dim) or 
            (batch_size, seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Get positional embeddings for required length
        pe = self.weight[:seq_len, :]
        
        # Add batch dimension if requested
        if batch_size is not None:
            pe = np.tile(pe[np.newaxis, :, :], (batch_size, 1, 1))
        
        return pe
    
    def backward(self, seq_len: int, grad_output: np.ndarray, 
                 batch_size: Optional[int] = None) -> None:
        """
        Backward pass: accumulate gradients.
        
        Args:
            seq_len: Sequence length used in forward pass
            grad_output: Gradient from upstream
            batch_size: Batch size used in forward pass
        """
        # Handle batch dimension
        if batch_size is not None and grad_output.ndim == 3:
            # Sum gradients across batch dimension
            grad_output = np.sum(grad_output, axis=0)
        
        # Accumulate gradients for used positions
        self.weight_grad[:seq_len, :] += grad_output
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters dictionary."""
        return {'weight': self.weight}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients dictionary."""
        return {'weight': self.weight_grad}

# =============================================================================
# COMBINED EMBEDDING LAYER
# =============================================================================

class TransformerEmbedding:
    """
    Combined token and positional embeddings for transformer input.
    
    This layer combines token embeddings with positional information to
    create the final input embeddings for the transformer. Based on
    experience with modular design from FPGA implementation work.
    
    The output is: token_embeddings + positional_embeddings
    
    Optional features:
    - Dropout for regularization
    - Layer normalization
    - Scaling of embeddings
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 max_seq_len: int,
                 padding_idx: Optional[int] = None,
                 pos_encoding_type: str = 'sinusoidal',
                 dropout: float = 0.1,
                 layer_norm: bool = False,
                 scale_embeddings: bool = True,
                 init_std: float = 0.02,
                 dtype: np.dtype = np.float32):
        """
        Initialize combined embedding layer.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            max_seq_len: Maximum sequence length
            padding_idx: Padding token index
            pos_encoding_type: 'sinusoidal' or 'learnable'
            dropout: Dropout probability
            layer_norm: Whether to apply layer normalization
            scale_embeddings: Whether to scale embeddings by sqrt(embed_dim)
            init_std: Initialization standard deviation
            dtype: Data type
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.scale_embeddings = scale_embeddings
        self.dtype = dtype
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
            init_std=init_std,
            dtype=dtype
        )
        
        # Positional encoding/embeddings
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                dtype=dtype
            )
            self.learnable_pos = False
        elif pos_encoding_type == 'learnable':
            self.pos_encoding = LearnablePositionalEmbedding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                init_std=init_std,
                dtype=dtype
            )
            self.learnable_pos = True
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")
        
        # Optional layer normalization
        if self.layer_norm:
            from layer_norm_py import LayerNorm
            self.norm = LayerNorm(embed_dim, dtype=dtype)
        
        # Scaling factor for embeddings (from original Transformer paper)
        self.embed_scale = math.sqrt(embed_dim) if scale_embeddings else 1.0
        
        print(f"TransformerEmbedding initialized:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - embed_dim: {embed_dim}")
        print(f"  - max_seq_len: {max_seq_len}")
        print(f"  - pos_encoding_type: {pos_encoding_type}")
        print(f"  - dropout: {dropout}")
        print(f"  - layer_norm: {layer_norm}")
        print(f"  - scale_embeddings: {scale_embeddings}")
    
    def forward(self, token_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass: compute combined embeddings.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embedding.forward(token_ids)
        
        # Scale embeddings if requested
        if self.scale_embeddings:
            token_embeds = token_embeds * self.embed_scale
        
        # Get positional encodings
        pos_embeds = self.pos_encoding.forward(seq_len, batch_size)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        
        # Apply layer normalization if enabled
        if self.layer_norm:
            embeddings = self.norm.forward(embeddings)
        
        # Apply dropout if in training mode
        if training and self.dropout > 0.0:
            embeddings = self._apply_dropout(embeddings)
        
        return embeddings
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout to embeddings.
        
        Args:
            x: Input embeddings
            
        Returns:
            Embeddings with dropout applied
        """
        # Create dropout mask
        keep_prob = 1.0 - self.dropout
        mask = np.random.binomial(1, keep_prob, x.shape).astype(self.dtype)
        
        # Apply dropout and scale
        return x * mask / keep_prob
    
    def backward(self, token_ids: np.ndarray, grad_output: np.ndarray,
                 training: bool = True) -> None:
        """
        Backward pass: compute gradients.
        
        Args:
            token_ids: Token IDs from forward pass
            grad_output: Gradient from upstream layers
            training: Whether in training mode
        """
        batch_size, seq_len = token_ids.shape
        
        # Apply layer norm backward if enabled
        if self.layer_norm:
            grad_output = self.norm.backward(grad_output)
        
        # Scale gradients if embeddings were scaled
        if self.scale_embeddings:
            grad_output = grad_output * self.embed_scale
        
        # Backward through token embeddings
        self.token_embedding.backward(token_ids, grad_output)
        
        # Backward through positional embeddings (if learnable)
        if self.learnable_pos:
            self.pos_encoding.backward(seq_len, grad_output, batch_size)
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all parameters."""
        params = {}
        
        # Token embedding parameters
        token_params = self.token_embedding.get_parameters()
        for key, value in token_params.items():
            params[f'token_embedding.{key}'] = value
        
        # Positional embedding parameters (if learnable)
        if self.learnable_pos:
            pos_params = self.pos_encoding.get_parameters()
            for key, value in pos_params.items():
                params[f'pos_encoding.{key}'] = value
        
        # Layer norm parameters (if enabled)
        if self.layer_norm:
            norm_params = self.norm.get_parameters()
            for key, value in norm_params.items():
                params[f'norm.{key}'] = value
        
        return params
    
    def set_parameters(self, params: dict) -> None:
        """Set all parameters from loaded checkpoint"""
        # Set token embedding parameters
        token_params = {k.replace('token_embedding.', ''): v for k, v in params.items() if k.startswith('token_embedding.')}
        if token_params:
            self.token_embedding.set_parameters(token_params)
        
        # Set positional embedding parameters (if learnable)
        if self.learnable_pos:
            pos_params = {k.replace('pos_encoding.', ''): v for k, v in params.items() if k.startswith('pos_encoding.')}
            if pos_params:
                self.pos_encoding.set_parameters(pos_params)
        
        # Set layer norm parameters (if enabled)
        if self.layer_norm:
            norm_params = {k.replace('norm.', ''): v for k, v in params.items() if k.startswith('norm.')}
            if norm_params:
                self.norm.set_parameters(norm_params)
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get all gradients."""
        grads = {}
        
        # Token embedding gradients
        token_grads = self.token_embedding.get_gradients()
        for key, value in token_grads.items():
            grads[f'token_embedding.{key}'] = value
        
        # Positional embedding gradients (if learnable)
        if self.learnable_pos:
            pos_grads = self.pos_encoding.get_gradients()
            for key, value in pos_grads.items():
                grads[f'pos_encoding.{key}'] = value
        
        # Layer norm gradients (if enabled)
        if self.layer_norm:
            norm_grads = self.norm.get_gradients()
            for key, value in norm_grads.items():
                grads[f'norm.{key}'] = value
        
        return grads

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_embedding_layer(config: Dict[str, Any], 
                         padding_idx: Optional[int] = None) -> TransformerEmbedding:
    """
    Factory function to create embedding layer from config.
    
    Args:
        config: Configuration dictionary
        padding_idx: Padding token index
        
    Returns:
        TransformerEmbedding instance
    """
    return TransformerEmbedding(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        max_seq_len=config['max_seq_len'],
        padding_idx=padding_idx,
        pos_encoding_type=config.get('pos_encoding_type', 'sinusoidal'),
        dropout=config.get('dropout', 0.1),
        layer_norm=config.get('embedding_layer_norm', False),
        scale_embeddings=config.get('scale_embeddings', True),
        init_std=PRECISION_CONFIG.get('init_std', 0.02),
        dtype=PRECISION_CONFIG.get('dtype', np.float32)
    )

def visualize_positional_encoding(embed_dim: int = 128, max_seq_len: int = 100) -> None:
    """
    Visualize positional encoding patterns for debugging/understanding.
    
    Args:
        embed_dim: Embedding dimension
        max_seq_len: Maximum sequence length to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return
    
    # Create positional encoding
    pos_enc = PositionalEncoding(embed_dim, max_seq_len)
    pe_matrix = pos_enc.forward(max_seq_len)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(pe_matrix.T, aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar()
    plt.title('Sinusoidal Positional Encoding')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.tight_layout()
    plt.show()
    
    print(f"Visualized positional encoding: {embed_dim}D x {max_seq_len} positions")

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING LAYERS TESTING")
    print("=" * 60)
    
    # Test parameters
    vocab_size = 1000
    embed_dim = 128
    max_seq_len = 50
    batch_size = 4
    seq_len = 20
    
    # Create sample token IDs
    np.random.seed(42)
    sample_token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nTest parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    
    # Test Token Embedding
    print(f"\n1. TESTING TOKEN EMBEDDING")
    print("-" * 40)
    
    token_emb = TokenEmbedding(vocab_size, embed_dim, padding_idx=0)
    token_output = token_emb.forward(sample_token_ids)
    
    print(f"Token embedding output shape: {token_output.shape}")
    print_array_stats(token_output, "Token embeddings")
    
    # Test Positional Encoding
    print(f"\n2. TESTING POSITIONAL ENCODING")
    print("-" * 40)
    
    pos_enc = PositionalEncoding(embed_dim, max_seq_len)
    pos_output = pos_enc.forward(seq_len, batch_size)
    
    print(f"Positional encoding output shape: {pos_output.shape}")
    print_array_stats(pos_output, "Positional encoding")
    
    # Test Learnable Positional Embedding
    print(f"\n3. TESTING LEARNABLE POSITIONAL EMBEDDING")
    print("-" * 40)
    
    learnable_pos = LearnablePositionalEmbedding(embed_dim, max_seq_len)
    learnable_output = learnable_pos.forward(seq_len, batch_size)
    
    print(f"Learnable positional embedding shape: {learnable_output.shape}")
    print_array_stats(learnable_output, "Learnable positional")
    
    # Test Combined Embedding (Sinusoidal)
    print(f"\n4. TESTING COMBINED EMBEDDING (SINUSOIDAL)")
    print("-" * 40)
    
    combined_emb_sin = TransformerEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        padding_idx=0,
        pos_encoding_type='sinusoidal',
        dropout=0.1,
        layer_norm=False,
        scale_embeddings=True
    )
    
    combined_output_sin = combined_emb_sin.forward(sample_token_ids, training=False)
    
    print(f"Combined embedding (sin) shape: {combined_output_sin.shape}")
    print_array_stats(combined_output_sin, "Combined embeddings (sinusoidal)")
    
    # Test Combined Embedding (Learnable)
    print(f"\n5. TESTING COMBINED EMBEDDING (LEARNABLE)")
    print("-" * 40)
    
    combined_emb_learn = TransformerEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        padding_idx=0,
        pos_encoding_type='learnable',
        dropout=0.0,  # Disable for testing
        layer_norm=True,
        scale_embeddings=True
    )
    
    combined_output_learn = combined_emb_learn.forward(sample_token_ids, training=False)
    
    print(f"Combined embedding (learnable) shape: {combined_output_learn.shape}")
    print_array_stats(combined_output_learn, "Combined embeddings (learnable)")
    
    # Test parameter counting
    print(f"\n6. PARAMETER ANALYSIS")
    print("-" * 40)
    
    # Count parameters
    sin_params = combined_emb_sin.get_parameters()
    learn_params = combined_emb_learn.get_parameters()
    
    sin_total = sum(p.size for p in sin_params.values())
    learn_total = sum(p.size for p in learn_params.values())
    
    print(f"Sinusoidal embedding parameters:")
    for name, param in sin_params.items():
        print(f"  {name}: {param.shape} ({param.size:,} params)")
    print(f"  Total: {sin_total:,} parameters")
    
    print(f"\nLearnable embedding parameters:")
    for name, param in learn_params.items():
        print(f"  {name}: {param.shape} ({param.size:,} params)")
    print(f"  Total: {learn_total:,} parameters")
    
    print(f"\nParameter difference: {learn_total - sin_total:,}")
    
    # Test gradient computation
    print(f"\n7. GRADIENT TESTING")
    print("-" * 40)
    
    # Create dummy gradient
    grad_shape = combined_output_sin.shape
    dummy_grad = np.random.randn(*grad_shape).astype(np.float32) * 0.01
    
    # Backward pass
    combined_emb_sin.backward(sample_token_ids, dummy_grad, training=False)
    combined_emb_learn.backward(sample_token_ids, dummy_grad, training=False)
    
    # Check gradients
    sin_grads = combined_emb_sin.get_gradients()
    learn_grads = combined_emb_learn.get_gradients()
    
    print("Gradient shapes (sinusoidal):")
    for name, grad in sin_grads.items():
        print(f"  {name}: {grad.shape}")
        check_finite(grad, name)
    
    print("Gradient shapes (learnable):")
    for name, grad in learn_grads.items():
        print(f"  {name}: {grad.shape}")
        check_finite(grad, name)
    
    print("\n" + "=" * 60)
    print("All embedding tests completed successfully!")
    print("Ready for integration with transformer architecture!")
    print("=" * 60)