#!/usr/bin/env python3
"""
Text Generation Example for Pure Python Transformer

This script loads a trained transformer model and generates text
based on a given prompt. Demonstrates autoregressive text generation.

Usage:
    python examples/generate_text.py "Once upon a time"
    python examples/generate_text.py --model-path model.npz --prompt "Hello world"

Author: Joel Oliver
"""

import sys
import os
import argparse
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
from config_py import MODEL_CONFIG
from transformer import Transformer
from tokenizer_py import CharacterTokenizer

def generate_text(model_path: str, tokenizer_path: str = None, 
                 prompt: str = "Once upon a time", max_length: int = 100,
                 temperature: float = 0.8, num_samples: int = 1):
    """
    Generate text using a trained transformer model
    
    Args:
        model_path: Path to saved model checkpoint
        tokenizer_path: Path to saved tokenizer (optional)
        prompt: Starting text prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        num_samples: Number of different samples to generate
    """
    print("=" * 60)
    print("Pure Python Transformer - Text Generation")
    print("=" * 60)
    
    # Load model configuration (matching the training script)
    config = MODEL_CONFIG.copy()
    config.update({
        'vocab_size': 1000,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'ff_dim': 256,
        'max_seq_len': 64,
        'dropout': 0.0,
        'attention_dropout': 0.0,
        'ff_dropout': 0.0
    })
    
    # Create and load model
    print(f"Loading model from {model_path}...")
    try:
        model = Transformer(config)
        model.load_checkpoint(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train a model first using examples/train_simple.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load tokenizer (same as used for training)
    print("Loading tokenizer...")
    try:
        tokenizer = CharacterTokenizer()
        tokenizer.load_vocab('simple_tokenizer.vocab')
        print("Tokenizer loaded successfully!")
    except FileNotFoundError:
        print("Tokenizer file not found. Creating a basic tokenizer...")
        tokenizer = CharacterTokenizer()
        # For demo purposes, create a basic vocabulary
        sample_text = """
        Once upon a time, in a land far away, there lived a wise old wizard.
        The wizard had a magical book that contained all the knowledge of the world.
        Every day, people from distant villages would come to seek his wisdom.
        He would read from his magical book and share stories of adventure.
        The stories were filled with heroes and dragons, magic and mystery.
        Children loved to hear about brave knights and clever princesses.
        The wizard's book never ran out of new tales to tell.
        And so the tradition continued, passed down through generations.
        """
        tokenizer.train([sample_text])
    
    print(f"\nGenerating text with:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max length: {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Number of samples: {num_samples}")
    print()
    
    # Generate multiple samples
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print("-" * 40)
        
        try:
            generated = model.generate(
                prompt=prompt,
                tokenizer=tokenizer,
                max_length=max_length,
                temperature=temperature
            )
            print(generated)
            
        except Exception as e:
            print(f"Error during generation: {e}")
        
        print()
    
    print("Generation complete!")

def interactive_generation(model_path: str):
    """Interactive text generation session"""
    print("=" * 60)
    print("Interactive Text Generation Session")
    print("Enter prompts and get generated text!")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 60)
    
    # Load model
    config = MODEL_CONFIG.copy()
    model = Transformer(config)
    
    try:
        model.load_checkpoint(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    sample_text = "Once upon a time, in a land far away, there lived a wise old wizard."
    tokenizer.train([sample_text])
    
    # Interactive loop
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'help':
                print("Commands:")
                print("  quit - Exit the session")
                print("  help - Show this help")
                print("  Any other text will be used as a generation prompt")
                continue
            elif not prompt:
                continue
            
            # Get generation parameters
            try:
                max_length = int(input("Max length (default 50): ") or "50")
                temperature = float(input("Temperature (default 0.8): ") or "0.8")
            except ValueError:
                max_length = 50
                temperature = 0.8
            
            print("\nGenerating...")
            generated = model.generate(prompt, tokenizer, max_length, temperature)
            print(f"\nGenerated text:\n{generated}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate text with transformer model")
    parser.add_argument("prompt", nargs="?", default="Once upon a time",
                       help="Text prompt for generation")
    parser.add_argument("--model-path", default="simple_model.npz",
                       help="Path to saved model checkpoint")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum length to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples to generate")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive generation session")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_generation(args.model_path)
    else:
        generate_text(
            model_path=args.model_path,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()