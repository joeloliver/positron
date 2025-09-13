#!/usr/bin/env python3
"""
Complete Walkthrough of Pure Python Transformer

This script demonstrates every step of the transformer process in detail,
showing exactly what happens to the data at each stage.

Author: Joel Oliver
"""

import sys
import numpy as np
from config_py import MODEL_CONFIG, TRAINING_CONFIG
from transformer import Transformer
from tokenizer_py import SimpleTokenizer
import matplotlib.pyplot as plt

def print_step(step_num, title, details=""):
    """Pretty print each step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")
    if details:
        print(details)

def show_array_info(name, array, max_elements=10):
    """Show detailed array information"""
    if isinstance(array, np.ndarray):
        print(f"\n📊 {name}:")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        print(f"  Min/Max: {array.min():.4f} / {array.max():.4f}")
        print(f"  Mean/Std: {array.mean():.4f} / {array.std():.4f}")
        if array.size <= max_elements:
            print(f"  Values: {array}")
        else:
            flat = array.flatten()
            print(f"  First {max_elements} values: {flat[:max_elements]}")
    else:
        print(f"\n📊 {name}: {array}")

def main():
    print("🧠 PURE PYTHON TRANSFORMER WALKTHROUGH")
    print("Understanding every step of the process\n")
    
    # =========================================================================
    # STEP 1: INPUT PREPARATION
    # =========================================================================
    
    print_step(1, "INPUT PREPARATION", "Converting human text to machine-readable format")
    
    # Sample input text
    input_text = "Once upon a time, there was a wise wizard."
    print(f"Input text: '{input_text}'")
    print(f"Text length: {len(input_text)} characters")
    
    # Initialize tokenizer
    print("\n🔤 Creating tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.train([input_text])
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}")
    
    # =========================================================================
    # STEP 2: TOKENIZATION
    # =========================================================================
    
    print_step(2, "TOKENIZATION", "Breaking text into tokens the model can understand")
    
    # Encode text to token IDs
    token_ids = tokenizer.encode(input_text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    # Show character-to-token mapping
    print("\n📝 Character-to-token mapping:")
    for i, char in enumerate(input_text[:20]):  # Show first 20 chars
        if i < len(token_ids):
            print(f"  '{char}' → {token_ids[i]}")
    
    # Prepare input for model (add batch dimension)
    input_ids = np.array([token_ids])  # Shape: (1, seq_len)
    show_array_info("Input token IDs", input_ids)
    
    # =========================================================================
    # STEP 3: MODEL CREATION
    # =========================================================================
    
    print_step(3, "MODEL CREATION", "Building the transformer architecture")
    
    # Create small model for demonstration
    config = MODEL_CONFIG.copy()
    config.update({
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 64,    # Smaller for clearer demonstration
        'num_heads': 4,
        'num_layers': 2,
        'ff_dim': 128,
        'max_seq_len': 64
    })
    
    print(f"Model configuration:")
    for key, value in config.items():
        if key in ['vocab_size', 'embed_dim', 'num_heads', 'num_layers', 'ff_dim']:
            print(f"  {key}: {value}")
    
    model = Transformer(config)
    print(f"\n✅ Model created with {model.get_num_parameters():,} parameters")
    
    # =========================================================================
    # STEP 4: EMBEDDING LAYER
    # =========================================================================
    
    print_step(4, "EMBEDDING LAYER", "Converting tokens to dense vectors with positional information")
    
    # Forward pass through embedding layer
    print("🔄 Processing through embedding layer...")
    embeddings = model.embedding.forward(input_ids)
    
    show_array_info("Token embeddings", embeddings)
    print(f"\n💡 Each token is now a {config['embed_dim']}-dimensional vector")
    print(f"   This gives the model rich semantic representation")
    
    # =========================================================================
    # STEP 5: TRANSFORMER LAYERS
    # =========================================================================
    
    print_step(5, "TRANSFORMER LAYERS", "Multi-head attention and feed-forward processing")
    
    print(f"🔄 Processing through {config['num_layers']} transformer layers...")
    
    # Track activations through layers
    activations = [embeddings]
    current_input = embeddings
    
    for layer_idx, layer in enumerate(model.layers):
        print(f"\n--- Layer {layer_idx + 1} ---")
        
        # Pre-norm
        norm1_out = layer.norm1.forward(current_input)
        print(f"After layer norm 1: mean={norm1_out.mean():.4f}, std={norm1_out.std():.4f}")
        
        # Multi-head attention
        print("🎯 Multi-head attention:")
        attn_out, attn_weights = layer.attention.forward(norm1_out)
        print(f"  Attention output shape: {attn_out.shape}")
        if attn_weights is not None:
            print(f"  Attention weights shape: {attn_weights.shape}")
            # Handle different attention weight shapes
            if len(attn_weights.shape) == 4:
                print(f"  Attention weights (first few): {attn_weights[0, 0, :5, :5]}")
            elif len(attn_weights.shape) == 3:
                print(f"  Attention weights (first few): {attn_weights[0, :5, :5]}")
            else:
                print(f"  Attention weights (first few): {attn_weights[:5, :5]}")
        
        # Residual connection
        current_input = current_input + attn_out
        print(f"After residual 1: mean={current_input.mean():.4f}, std={current_input.std():.4f}")
        
        # Second norm
        norm2_out = layer.norm2.forward(current_input)
        
        # Feed-forward network
        print("🧠 Feed-forward network:")
        ff_out = layer.feed_forward.forward(norm2_out)
        print(f"  FF output shape: {ff_out.shape}")
        print(f"  FF output mean/std: {ff_out.mean():.4f} / {ff_out.std():.4f}")
        
        # Second residual connection
        current_input = current_input + ff_out
        print(f"After residual 2: mean={current_input.mean():.4f}, std={current_input.std():.4f}")
        
        activations.append(current_input.copy())
    
    # =========================================================================
    # STEP 6: OUTPUT PROJECTION
    # =========================================================================
    
    print_step(6, "OUTPUT PROJECTION", "Converting to vocabulary logits")
    
    # Final layer normalization
    final_norm_out = model.final_norm.forward(current_input)
    print(f"After final layer norm: mean={final_norm_out.mean():.4f}, std={final_norm_out.std():.4f}")
    
    # Project to vocabulary
    logits = np.matmul(final_norm_out, model.output_projection)
    if model.output_bias is not None:
        logits += model.output_bias
    
    show_array_info("Final logits", logits)
    print(f"\n💡 Logits represent the model's 'confidence' for each token in vocabulary")
    
    # =========================================================================
    # STEP 7: PROBABILITY DISTRIBUTION
    # =========================================================================
    
    print_step(7, "PROBABILITY DISTRIBUTION", "Converting logits to probabilities")
    
    # Apply softmax to get probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    # Get probabilities for the last token (next token prediction)
    last_token_logits = logits[0, -1, :]  # Last position, all vocab
    probabilities = softmax(last_token_logits)
    
    print(f"Probability distribution over vocabulary ({len(probabilities)} tokens)")
    
    # Show top predictions
    top_indices = np.argsort(probabilities)[-5:][::-1]  # Top 5
    print(f"\n🏆 Top 5 predictions:")
    for i, idx in enumerate(top_indices):
        char = tokenizer.decode([idx]) if idx < len(tokenizer.vocab) else '<UNK>'
        print(f"  {i+1}. '{char}' (token {idx}): {probabilities[idx]:.4f}")
    
    # =========================================================================
    # STEP 8: TEXT GENERATION
    # =========================================================================
    
    print_step(8, "TEXT GENERATION", "Sampling the next token")
    
    # Sample next token
    next_token = np.random.choice(len(probabilities), p=probabilities)
    next_char = tokenizer.decode([next_token])
    
    print(f"Sampled next token: {next_token} → '{next_char}'")
    print(f"Probability of sampled token: {probabilities[next_token]:.4f}")
    
    # Show complete generation
    generated_text = input_text + next_char
    print(f"\n📝 Generated text: '{generated_text}'")
    
    # =========================================================================
    # STEP 9: TRAINING INSIGHTS
    # =========================================================================
    
    print_step(9, "TRAINING INSIGHTS", "How the model learns")
    
    print(f"🎯 During training, the model:")
    print(f"  1. Takes input text: '{input_text}'")
    print(f"  2. Predicts next character at each position")
    print(f"  3. Compares predictions with actual next characters")
    print(f"  4. Adjusts {model.get_num_parameters():,} parameters to improve")
    print(f"  5. Gradually learns patterns in language")
    
    # Show parameter distribution
    all_params = []
    all_params.extend(model.output_projection.flatten())
    if model.output_bias is not None:
        all_params.extend(model.output_bias.flatten())
    all_params = np.array(all_params)
    
    print(f"\n📊 Parameter statistics:")
    print(f"  Total parameters: {len(all_params):,}")
    print(f"  Parameter range: {all_params.min():.4f} to {all_params.max():.4f}")
    print(f"  Parameter mean/std: {all_params.mean():.4f} / {all_params.std():.4f}")
    
    # =========================================================================
    # STEP 10: ATTENTION VISUALIZATION
    # =========================================================================
    
    print_step(10, "ATTENTION VISUALIZATION", "What the model focuses on")
    
    if len(model.layers) > 0 and hasattr(model.layers[0], 'attention'):
        print(f"🔍 The attention mechanism allows the model to:")
        print(f"  • Focus on relevant parts of the input")
        print(f"  • Understand relationships between tokens")
        print(f"  • Process sequences of different lengths")
        print(f"  • Capture long-range dependencies")
        
        print(f"\n💡 With {config['num_heads']} attention heads, the model can:")
        print(f"  • Learn different types of relationships simultaneously")
        print(f"  • Some heads might focus on syntax")
        print(f"  • Others might focus on semantics")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print_step("✅", "COMPLETE PIPELINE SUMMARY")
    
    print(f"🔄 Data Flow Summary:")
    print(f"  Text → Tokens → Embeddings → Attention → FFN → Logits → Probabilities")
    print(f"  {len(input_text)} chars → {len(token_ids)} tokens → {embeddings.shape} → {logits.shape} → {len(probabilities)} probs")
    
    print(f"\n🧠 Key Insights:")
    print(f"  • Pure Python/NumPy implementation - no ML frameworks!")
    print(f"  • Every operation is mathematically transparent")
    print(f"  • {model.get_num_parameters():,} learnable parameters")
    print(f"  • Attention mechanism enables sequence understanding")
    print(f"  • Feed-forward networks provide non-linear processing")
    print(f"  • Layer normalization ensures stable training")
    
    print(f"\n🚀 This foundation can:")
    print(f"  • Generate coherent text")
    print(f"  • Complete sentences and stories")
    print(f"  • Learn patterns in any text domain")
    print(f"  • Scale to larger models and datasets")
    
    print(f"\n{'='*60}")
    print("WALKTHROUGH COMPLETE! 🎉")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()