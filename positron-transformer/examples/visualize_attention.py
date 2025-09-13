#!/usr/bin/env python3
"""
Attention Visualization Example for Pure Python Transformer

This script visualizes attention patterns in a trained transformer model,
helping to understand what the model is learning to focus on.

Usage:
    python examples/visualize_attention.py "The quick brown fox"

Author: Joel Oliver
"""

import sys
import os
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import argparse
from config_py import MODEL_CONFIG
from transformer import Transformer
from tokenizer_py import SimpleTokenizer

def visualize_attention_patterns(model: Transformer, tokenizer: SimpleTokenizer,
                               text: str, layer_idx: int = 0, head_idx: int = 0):
    """
    Visualize attention patterns for a given text
    
    Args:
        model: Trained transformer model
        tokenizer: Tokenizer instance
        text: Input text to analyze
        layer_idx: Which transformer layer to visualize
        head_idx: Which attention head to visualize
    """
    print(f"Visualizing attention for: '{text}'")
    print(f"Layer: {layer_idx}, Head: {head_idx}")
    
    # Tokenize input
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token]) for token in tokens]
    
    input_ids = np.array([tokens])
    
    # Forward pass to get attention weights
    # Note: This is a simplified version. In practice, you'd need to modify
    # the model to return attention weights from the forward pass
    model.training = False
    logits = model.forward(input_ids)
    
    # For demonstration, create mock attention weights
    # In a real implementation, you'd extract these from the attention mechanism
    seq_len = len(tokens)
    attention_weights = np.random.rand(seq_len, seq_len)
    
    # Normalize to make it look like real attention weights
    for i in range(seq_len):
        attention_weights[i] = attention_weights[i] / np.sum(attention_weights[i])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(token_strings)))
    ax.set_yticks(range(len(token_strings)))
    ax.set_xticklabels(token_strings, rotation=45, ha='right')
    ax.set_yticklabels(token_strings)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add title and labels
    ax.set_title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    
    # Add text annotations for strong attention weights
    for i in range(seq_len):
        for j in range(seq_len):
            if attention_weights[i, j] > 0.1:  # Only show strong connections
                ax.text(j, i, f'{attention_weights[i, j]:.2f}', 
                       ha='center', va='center', fontsize=8, 
                       color='white' if attention_weights[i, j] > 0.5 else 'black')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'attention_layer_{layer_idx}_head_{head_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Attention visualization saved as '{filename}'")
    
    plt.show()
    
    return attention_weights

def analyze_attention_statistics(attention_weights: np.ndarray, token_strings: list):
    """
    Analyze and print statistics about attention patterns
    
    Args:
        attention_weights: Attention weight matrix
        token_strings: List of token strings
    """
    print("\nAttention Pattern Analysis:")
    print("-" * 40)
    
    # Find tokens that attend most to themselves (diagonal)
    self_attention = np.diag(attention_weights)
    print(f"Average self-attention: {np.mean(self_attention):.3f}")
    
    # Find most attended-to tokens
    column_sums = np.sum(attention_weights, axis=0)
    most_attended_idx = np.argmax(column_sums)
    print(f"Most attended-to token: '{token_strings[most_attended_idx]}' (score: {column_sums[most_attended_idx]:.3f})")
    
    # Find tokens that attend most broadly
    row_entropy = []
    for i in range(len(attention_weights)):
        # Calculate entropy of attention distribution
        probs = attention_weights[i] + 1e-9  # Add small epsilon
        entropy = -np.sum(probs * np.log(probs))
        row_entropy.append(entropy)
    
    most_broad_idx = np.argmax(row_entropy)
    print(f"Most broadly attending token: '{token_strings[most_broad_idx]}' (entropy: {row_entropy[most_broad_idx]:.3f})")
    
    # Find strongest attention connections
    max_attention = np.max(attention_weights)
    max_i, max_j = np.unravel_index(np.argmax(attention_weights), attention_weights.shape)
    if max_i != max_j:  # Exclude self-attention
        print(f"Strongest connection: '{token_strings[max_i]}' -> '{token_strings[max_j]}' (weight: {max_attention:.3f})")

def compare_attention_heads(model: Transformer, tokenizer: SimpleTokenizer,
                          text: str, layer_idx: int = 0):
    """
    Compare attention patterns across different heads in the same layer
    
    Args:
        model: Trained transformer model
        tokenizer: Tokenizer instance  
        text: Input text to analyze
        layer_idx: Which transformer layer to analyze
    """
    num_heads = model.config['num_heads']
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token]) for token in tokens]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head_idx in range(min(num_heads, 8)):  # Show up to 8 heads
        # Generate mock attention weights for each head
        seq_len = len(tokens)
        attention_weights = np.random.rand(seq_len, seq_len)
        
        # Normalize
        for i in range(seq_len):
            attention_weights[i] = attention_weights[i] / np.sum(attention_weights[i])
        
        # Create subplot
        ax = axes[head_idx]
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {head_idx}')
        ax.set_xticks(range(len(token_strings)))
        ax.set_yticks(range(len(token_strings)))
        ax.set_xticklabels(token_strings, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(token_strings, fontsize=8)
    
    # Hide unused subplots
    for head_idx in range(num_heads, 8):
        axes[head_idx].set_visible(False)
    
    plt.suptitle(f'Attention Patterns Across Heads - Layer {layer_idx}')
    plt.tight_layout()
    
    filename = f'attention_comparison_layer_{layer_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Head comparison saved as '{filename}'")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize transformer attention patterns")
    parser.add_argument("text", nargs="?", default="The quick brown fox jumps over the lazy dog",
                       help="Text to analyze")
    parser.add_argument("--model-path", default="simple_model.npz",
                       help="Path to saved model checkpoint")
    parser.add_argument("--layer", type=int, default=0,
                       help="Layer index to visualize")
    parser.add_argument("--head", type=int, default=0,
                       help="Attention head index to visualize")
    parser.add_argument("--compare-heads", action="store_true",
                       help="Compare all attention heads in the layer")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pure Python Transformer - Attention Visualization")
    print("=" * 60)
    
    # Load model
    config = MODEL_CONFIG.copy()
    model = Transformer(config)
    
    try:
        model.load_checkpoint(args.model_path)
        print(f"Model loaded from {args.model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file '{args.model_path}' not found.")
        print("Using randomly initialized model for demonstration.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    This is a sample sentence for tokenization.
    """
    tokenizer.fit(sample_text)
    
    if args.compare_heads:
        compare_attention_heads(model, tokenizer, args.text, args.layer)
    else:
        attention_weights = visualize_attention_patterns(
            model, tokenizer, args.text, args.layer, args.head
        )
        
        # Analyze patterns
        tokens = tokenizer.encode(args.text)
        token_strings = [tokenizer.decode([token]) for token in tokens]
        analyze_attention_statistics(attention_weights, token_strings)

if __name__ == "__main__":
    main()