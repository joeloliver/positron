"""
Complete Transformer Model Implementation

This module assembles all transformer components into a complete model,
implementing the full transformer architecture with multiple layers,
residual connections, and layer normalization.

Author: Joel Oliver
Based on: Pure Python Transformer Implementation from Scratch
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from config_py import MODEL_CONFIG
from attention_py import MultiHeadAttention
from feedforward_py import FeedForward as FeedForwardNetwork
from layer_norm_py import LayerNorm as LayerNormalization
from embeddings_py import TransformerEmbedding
from utils_py import xavier_uniform_init as initialize_weights, create_causal_mask as apply_causal_mask


class TransformerBlock:
    """
    Single Transformer block implementing:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Layer normalization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embed_dim = config['embed_dim']
        
        # Initialize components
        self.attention = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=config['num_heads'],
            dropout=config.get('attention_dropout', 0.1),
            bias=config.get('use_bias', True)
        )
        self.feed_forward = FeedForwardNetwork(
            embed_dim=self.embed_dim,
            ff_dim=config.get('ff_dim', self.embed_dim * 4),
            activation=config.get('activation', 'gelu'),
            dropout=config.get('ff_dropout', 0.1),
            bias=config.get('use_bias', True)
        )
        self.norm1 = LayerNormalization(self.embed_dim, config.get('layer_norm_eps', 1e-5))
        self.norm2 = LayerNormalization(self.embed_dim, config.get('layer_norm_eps', 1e-5))
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Pre-normalization architecture
        # Multi-head attention with residual connection
        norm1_out = self.norm1.forward(x)
        attn_out, _ = self.attention.forward(norm1_out, attn_mask=mask)
        x = x + attn_out  # Residual connection
        
        # Feed-forward network with residual connection
        norm2_out = self.norm2.forward(x)
        ff_out = self.feed_forward.forward(norm2_out)
        x = x + ff_out  # Residual connection
        
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through transformer block"""
        # Backward through second residual connection
        grad_ff = grad_output
        grad_residual2 = grad_output
        
        # Backward through feed-forward
        grad_norm2 = self.feed_forward.backward(grad_ff)
        grad_x2 = self.norm2.backward(grad_norm2)
        grad_x2 += grad_residual2  # Add residual gradient
        
        # Backward through first residual connection
        grad_attn = grad_x2
        grad_residual1 = grad_x2
        
        # Backward through attention
        grad_query, grad_key, grad_value = self.attention.backward(grad_attn)
        # For self-attention, query, key, and value all come from the same input
        grad_norm1 = grad_query + grad_key + grad_value
        grad_x1 = self.norm1.backward(grad_norm1)
        grad_x1 += grad_residual1  # Add residual gradient
        
        return grad_x1
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving"""
        params = {}
        
        # Attention parameters
        attn_params = self.attention.get_parameters()
        for key, value in attn_params.items():
            params[f'attention.{key}'] = value
        
        # Feed forward parameters
        ff_params = self.feed_forward.get_parameters()
        for key, value in ff_params.items():
            params[f'feed_forward.{key}'] = value
        
        # Layer norm parameters
        norm1_params = self.norm1.get_parameters()
        for key, value in norm1_params.items():
            params[f'norm1.{key}'] = value
            
        norm2_params = self.norm2.get_parameters()
        for key, value in norm2_params.items():
            params[f'norm2.{key}'] = value
        
        return params
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from loaded checkpoint"""
        # Set attention parameters
        attn_params = {k.replace('attention.', ''): v for k, v in params.items() if k.startswith('attention.')}
        if attn_params:
            self.attention.set_parameters(attn_params)
        
        # Set feed forward parameters
        ff_params = {k.replace('feed_forward.', ''): v for k, v in params.items() if k.startswith('feed_forward.')}
        if ff_params:
            self.feed_forward.set_parameters(ff_params)
        
        # Set layer norm parameters
        norm1_params = {k.replace('norm1.', ''): v for k, v in params.items() if k.startswith('norm1.')}
        if norm1_params:
            self.norm1.set_parameters(norm1_params)
            
        norm2_params = {k.replace('norm2.', ''): v for k, v in params.items() if k.startswith('norm2.')}
        if norm2_params:
            self.norm2.set_parameters(norm2_params)


class Transformer:
    """
    Complete Transformer model for autoregressive language modeling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = MODEL_CONFIG
        self.config = config
        
        # Model dimensions
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.num_layers = config['num_layers']
        self.max_seq_len = config['max_seq_len']
        
        # Initialize components
        self.embedding = TransformerEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            dropout=config.get('dropout', 0.1)
        )
        self.layers = [TransformerBlock(config) for _ in range(self.num_layers)]
        self.final_norm = LayerNormalization(self.embed_dim, config.get('layer_norm_eps', 1e-5))
        
        # Output projection to vocabulary
        self.output_projection = initialize_weights((self.embed_dim, self.vocab_size))
        self.output_bias = np.zeros(self.vocab_size) if config.get('use_bias', True) else None
        
        # Training state
        self.training = True
        
    def forward(self, input_ids: np.ndarray, 
                attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through the complete transformer
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask for autoregressive modeling
        if self.config.get('causal_mask', True):
            causal_mask = apply_causal_mask(seq_len)
            if attention_mask is not None:
                # Combine with provided attention mask
                causal_mask = causal_mask * attention_mask
        else:
            causal_mask = attention_mask
        
        # Store input_ids for backward pass
        self.input_ids = input_ids
        
        # Embedding layer
        x = self.embedding.forward(input_ids)
        
        # Store activations for backward pass
        self.activations = [x]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer.forward(x, mask=causal_mask)
            self.activations.append(x)
        
        # Final layer normalization
        x = self.final_norm.forward(x)
        
        # Output projection
        logits = np.matmul(x, self.output_projection)
        if self.output_bias is not None:
            logits += self.output_bias
        
        return logits
    
    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass through the complete transformer
        
        Args:
            grad_logits: Gradient of loss w.r.t. logits
        """
        # Gradient w.r.t. output projection
        x_final = self.activations[-1]  # Output of final layer norm
        
        grad_output_proj = np.matmul(x_final.transpose(0, 2, 1), grad_logits)
        self.grad_output_projection = grad_output_proj.sum(axis=0)
        
        if self.output_bias is not None:
            self.grad_output_bias = grad_logits.sum(axis=(0, 1))
        
        # Gradient w.r.t. final layer norm input
        grad_x = np.matmul(grad_logits, self.output_projection.T)
        
        # Backward through final layer normalization
        grad_x = self.final_norm.backward(grad_x)
        
        # Backward through transformer layers (in reverse order)
        for i in range(len(self.layers) - 1, -1, -1):
            grad_x = self.layers[i].backward(grad_x)
        
        # Backward through embedding
        self.embedding.backward(self.input_ids, grad_x)
    
    def generate(self, prompt: str, tokenizer, max_length: int = 100, 
                 temperature: float = 1.0) -> str:
        """
        Generate text autoregressively
        
        Args:
            prompt: Starting text
            tokenizer: Tokenizer instance
            max_length: Maximum length to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        generated_tokens = tokens.tolist()
        
        self.training = False  # Set to inference mode
        
        for _ in range(max_length):
            # Prepare input (last max_seq_len tokens)
            input_tokens = generated_tokens[-self.max_seq_len:]
            input_ids = np.array([input_tokens])
            
            # Forward pass
            logits = self.forward(input_ids)
            next_token_logits = logits[0, -1, :]  # Logits for next token
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample next token
            probs = self._softmax(next_token_logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            generated_tokens.append(next_token)
            
            # Check for end token (if using)
            if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
                break
        
        self.training = True  # Reset to training mode
        return tokenizer.decode(np.array(generated_tokens))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_num_parameters(self) -> int:
        """Count total number of parameters"""
        # Simplified version - just count output projection for now
        total_params = self.output_projection.size
        if self.output_bias is not None:
            total_params += self.output_bias.size
        return total_params
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint (legacy method - use CheckpointManager for better separation)"""
        from checkpoint import CheckpointManager
        import os
        
        # If filepath ends with .npz, extract directory and name
        if filepath.endswith('.npz'):
            directory = os.path.dirname(filepath) or '.'
            name = os.path.basename(filepath).replace('.npz', '')
        else:
            directory = filepath
            name = 'model'
        
        CheckpointManager.save(self, directory, name)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint (legacy method - use CheckpointManager for better separation)"""
        from checkpoint import CheckpointManager
        import os
        
        # Handle both old .npz files and new directory structure
        if filepath.endswith('.npz'):
            # Try to load new format first
            directory = os.path.dirname(filepath) or '.'
            name = os.path.basename(filepath).replace('.npz', '')
            try:
                CheckpointManager.load(self, directory, name)
            except:
                # Fall back to old format
                checkpoint = np.load(filepath, allow_pickle=True)
                
                # Load main model weights
                if 'output_projection' in checkpoint:
                    self.output_projection = checkpoint['output_projection']
                if 'output_bias' in checkpoint:
                    self.output_bias = checkpoint['output_bias']
                
                print(f"Loaded legacy checkpoint from {filepath}")
        else:
            # New format - directory based
            CheckpointManager.load(self, filepath, 'model')


if __name__ == "__main__":
    # Simple test
    config = MODEL_CONFIG.copy()
    config['vocab_size'] = 1000
    config['embed_dim'] = 128
    config['num_heads'] = 4
    config['num_layers'] = 2
    
    model = Transformer(config)
    print(f"Transformer model created with {model.get_num_parameters():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    logits = model.forward(input_ids)
    print(f"Forward pass successful: {logits.shape}")