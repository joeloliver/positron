#!/usr/bin/env python3
"""
Simple Training Example for Pure Python Transformer

This script demonstrates how to train a small transformer model
on sample text data. Perfect for learning and experimentation.

Usage:
    python examples/train_simple.py

Author: Joel Oliver
"""

import sys
import os
import argparse
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
from config_py import MODEL_CONFIG, TRAINING_CONFIG
from transformer import Transformer
from tokenizer_py import CharacterTokenizer
from training import train_model
import matplotlib.pyplot as plt
from model_config import get_model_config, TEST_TRAINING_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Simple Training Example for Pure Python Transformer')
    parser.add_argument('--clean', action='store_true', help='Remove old model files before training')
    parser.add_argument('--config', choices=['small', 'improved'], default='improved',
                        help='Model configuration to use (default: improved)')
    parser.add_argument('--long-training', action='store_true',
                        help='Use long training configuration (200 epochs)')
    args = parser.parse_args()

    # Clean old files if requested
    if args.clean:
        files_to_remove = ['simple_model.npz', 'simple_tokenizer.vocab', 'training_loss.png']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        print()

    print("=" * 60)
    print("Pure Python Transformer - Simple Training Example")
    print("=" * 60)
    
    # Use centralized configuration
    config = get_model_config(args.config)

    # Choose training configuration based on arguments
    if args.long_training:
        from model_config import DEFAULT_TRAINING_CONFIG
        train_config = DEFAULT_TRAINING_CONFIG.copy()
    else:
        train_config = TEST_TRAINING_CONFIG.copy()
    
    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load training text from file
    with open('data/sample/tiny_stories.txt', 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print("Creating tokenizer...")

    # Initialize tokenizer first
    tokenizer = CharacterTokenizer()
    tokenizer.train([sample_text])  # train expects a list of texts

    # Update vocab size based on actual tokenizer vocabulary
    config['vocab_size'] = tokenizer.vocab_size
    print(f"Updated vocab_size to {tokenizer.vocab_size}")

    print("Creating model...")
    # Create model with correct vocab size
    model = Transformer(config)
    # print(f"Model created with {model.get_num_parameters():,} parameters")
    
    print("\nStarting training...")
    
    # Train the model
    try:
        metrics = train_model(model, tokenizer, sample_text, train_config)
        
        print("\nTraining completed successfully!")
        print(f"Final training loss: {metrics['train_losses'][-1]:.4f}")
        
        # Plot training curves if matplotlib is available
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(metrics['train_losses'])
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure instead of showing
            print("Training curve saved as 'training_loss.png'")
        except Exception as e:
            print(f"Could not plot training curves: {e}")
        
        # Test text generation
        print("\nTesting text generation...")
        prompt = "Once upon a time"
        generated_text = model.generate(prompt, tokenizer, max_length=50, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        
        # Save model and tokenizer
        model.save_checkpoint('simple_model.npz')
        tokenizer.save_vocab('simple_tokenizer.vocab')
        print("\nModel saved as 'simple_model.npz'")
        print("Tokenizer saved as 'simple_tokenizer.vocab'")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()