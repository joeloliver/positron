"""
Basic tests for Pure Python Transformer Implementation

These tests verify that the core components can be imported
and basic operations work correctly.

Author: Joel Oliver
"""

import sys
import os
sys.path.append('..')

import pytest
import numpy as np
from config_py import MODEL_CONFIG
from tokenizer_py import SimpleTokenizer
from transformer import Transformer

def test_imports():
    """Test that all modules can be imported"""
    try:
        import config_py
        import tokenizer_py
        import embeddings_py
        import attention_py
        import feedforward_py
        import layer_norm_py
        import transformer
        import training
        import utils_py
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_model_creation():
    """Test that we can create a transformer model"""
    config = MODEL_CONFIG.copy()
    config.update({
        'vocab_size': 100,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2
    })
    
    model = Transformer(config)
    assert model is not None
    assert model.vocab_size == 100
    assert model.embed_dim == 64

def test_tokenizer_basic():
    """Test basic tokenizer functionality"""
    tokenizer = SimpleTokenizer()
    
    # Test fitting
    sample_text = "Hello world. This is a test."
    tokenizer.fit(sample_text)
    
    # Test encoding/decoding
    text = "Hello world"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert isinstance(decoded, str)

def test_model_forward_pass():
    """Test that model forward pass works"""
    config = MODEL_CONFIG.copy()
    config.update({
        'vocab_size': 100,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'max_seq_len': 32
    })
    
    model = Transformer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward pass
    logits = model.forward(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config['vocab_size'])
    assert logits.shape == expected_shape

def test_parameter_counting():
    """Test parameter counting functionality"""
    config = MODEL_CONFIG.copy()
    config.update({
        'vocab_size': 100,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2
    })
    
    model = Transformer(config)
    num_params = model.get_num_parameters()
    
    assert isinstance(num_params, int)
    assert num_params > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])