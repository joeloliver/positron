#!/usr/bin/env python3
"""
Text Generation with BPE Tokenization

This script generates text using a model trained with BPE tokenization.

Usage:
    python examples/generate_bpe.py "Once upon a time"
    python examples/generate_bpe.py --model-path bpe_model.npz --prompt "Hello world"

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
from model_config import get_inference_config

def generate_text(model_path: str, tokenizer_path: str = 'bpe_tokenizer.json',
                 prompt: str = "Once upon a time", max_length: int = 100,
                 temperature: float = 0.8, num_samples: int = 1,
                 config_name: str = 'improved'):
    """
    Generate text using a trained transformer model with BPE tokenization

    Args:
        model_path: Path to saved model checkpoint
        tokenizer_path: Path to saved BPE tokenizer
        prompt: Starting text prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        num_samples: Number of different samples to generate
        config_name: Model configuration name
    """
    print("=" * 60)
    print("Pure Python Transformer - BPE Text Generation")
    print("=" * 60)

    # Load model configuration from centralized source
    config = get_inference_config(config_name)

    # Load BPE tokenizer
    print(f"Loading BPE tokenizer from {tokenizer_path}...")
    try:
        tokenizer = BPETokenizer()
        tokenizer.load_vocab(tokenizer_path)
        print("BPE tokenizer loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Tokenizer file '{tokenizer_path}' not found.")
        print("Please train a model first using examples/train_bpe.py")
        return
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Update config with correct vocab size
    config['vocab_size'] = tokenizer.get_vocab_size()

    # Create and load model
    print(f"Loading model from {model_path}...")
    try:
        model = Transformer(config)
        model.load_checkpoint(model_path)
        print("Model loaded successfully!")
        print(f"Model has {model.get_num_parameters():,} parameters")
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train a model first using examples/train_bpe.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"\nGenerating text with:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max length: {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Number of samples: {num_samples}")
    print()

    # Generate samples
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        print("-" * 40)
        try:
            generated_text = model.generate(prompt, tokenizer, max_length, temperature)
            print(f"{generated_text}")
        except Exception as e:
            print(f"Generation failed: {e}")
        print()

    print("Generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate text with BPE tokenization")
    parser.add_argument("prompt", nargs="?", default="Once upon a time",
                       help="Text prompt for generation")
    parser.add_argument("--model-path", default="bpe_model.npz",
                       help="Path to saved model checkpoint")
    parser.add_argument("--tokenizer-path", default="bpe_tokenizer.json",
                       help="Path to saved BPE tokenizer")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum length to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples to generate")
    parser.add_argument("--config", choices=["small", "improved"], default="improved",
                       help="Model configuration to use (default: improved)")

    args = parser.parse_args()

    generate_text(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        num_samples=args.num_samples,
        config_name=args.config
    )

if __name__ == "__main__":
    main()