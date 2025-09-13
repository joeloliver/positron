#!/usr/bin/env python3
"""
Training Example with BPE Subword Tokenization

This script demonstrates training with BPE tokenization, which should
produce much better results than character-level tokenization.

Usage:
    python examples/train_bpe.py
    python examples/train_bpe.py --config small --vocab-size 4000

Author: Joel Oliver
"""

import sys
import os
import argparse
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
from transformer import Transformer
from subword_tokenizer import BPETokenizer
from training import train_model
import matplotlib.pyplot as plt
from model_config import get_model_config, TEST_TRAINING_CONFIG, DEFAULT_TRAINING_CONFIG, BPE_TRAINING_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Training with BPE Subword Tokenization')
    parser.add_argument('--clean', action='store_true', help='Remove old model files before training')
    parser.add_argument('--config', choices=['small', 'improved'], default='improved',
                        help='Model configuration to use (default: improved)')
    parser.add_argument('--long-training', action='store_true',
                        help='Use long training configuration (200 epochs)')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='BPE vocabulary size (default: 2000)')
    parser.add_argument('--data-file', default='data/sample/tiny_stories.txt',
                        help='Training data file')
    args = parser.parse_args()

    # Clean old files if requested
    if args.clean:
        files_to_remove = ['bpe_model.npz', 'bpe_tokenizer.json', 'training_loss.png']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        print()

    print("=" * 60)
    print("Pure Python Transformer - BPE Training Example")
    print("=" * 60)

    # Use centralized configuration
    config = get_model_config(args.config)

    # Choose training configuration based on arguments
    if args.long_training:
        train_config = DEFAULT_TRAINING_CONFIG.copy()
    else:
        train_config = BPE_TRAINING_CONFIG.copy()  # Use BPE-optimized config

    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load training text from file
    print(f"Loading training data from {args.data_file}...")
    try:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            sample_text = f.read()
        print(f"Loaded {len(sample_text)} characters of training data")
    except FileNotFoundError:
        print(f"Error: Training file '{args.data_file}' not found.")
        return

    print("Creating BPE tokenizer...")

    # Initialize BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train([sample_text])

    # Update vocab size based on actual tokenizer vocabulary
    config['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Updated vocab_size to {tokenizer.get_vocab_size()}")

    print("Creating model...")
    # Create model with correct vocab size
    model = Transformer(config)

    print(f"\nModel has {model.get_num_parameters():,} parameters")
    print(f"BPE vocabulary size: {tokenizer.get_vocab_size()}")

    print("\nStarting training...")

    # Train the model
    try:
        metrics = train_model(model, tokenizer, sample_text, train_config)

        print("\nTraining completed successfully!")
        print(f"Final training loss: {metrics['train_losses'][-1]:.4f}")

        # Plot training curves
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(metrics['train_losses'])
            plt.title('Training Loss (BPE Tokenization)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Training curve saved as 'training_loss.png'")
        except Exception as e:
            print(f"Could not plot training curves: {e}")

        # Test text generation
        print("\nTesting text generation...")
        prompt = "Once upon a time"
        try:
            generated_text = model.generate(prompt, tokenizer, max_length=50, temperature=0.8)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated_text}'")
        except Exception as e:
            print(f"Generation failed: {e}")

        # Save model and tokenizer
        model.save_checkpoint('bpe_model.npz')
        tokenizer.save_vocab('bpe_tokenizer.json')
        print("\nModel saved as 'bpe_model.npz'")
        print("Tokenizer saved as 'bpe_tokenizer.json'")

        # Show some example tokenizations
        print("\nExample tokenizations:")
        test_phrases = [
            "Once upon a time",
            "The quick brown fox jumps",
            "In the beginning there was"
        ]
        for phrase in test_phrases:
            tokens = tokenizer.encode(phrase, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            print(f"  '{phrase}' -> {tokens[:8]}... -> '{decoded}'")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()