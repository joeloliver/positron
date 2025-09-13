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
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
from config_py import MODEL_CONFIG, TRAINING_CONFIG
from transformer import Transformer
from tokenizer_py import CharacterTokenizer
from training import train_model
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Pure Python Transformer - Simple Training Example")
    print("=" * 60)
    
    # Create a small configuration for quick training
    config = MODEL_CONFIG.copy()
    config.update({
        'vocab_size': 1000,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'ff_dim': 256,
        'max_seq_len': 64,
        'dropout': 0.0,           # Disable dropout temporarily
        'attention_dropout': 0.0,  # Disable attention dropout
        'ff_dropout': 0.0         # Disable FF dropout
    })
    
    # Training configuration
    train_config = TRAINING_CONFIG.copy()
    train_config.update({
        'learning_rate': 5e-4,  # Lower learning rate for better convergence
        'batch_size': 8,
        'num_epochs': 50,  # Train much longer
        'seq_len': 32,
        'log_interval': 50  # Less frequent logging since more epochs
    })
    
    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Sample training text
    sample_text = """
    Once upon a time, in a land far away, there lived a wise old wizard.
    The wizard had a magical book that contained all the knowledge of the world.
    Every day, people from distant villages would come to seek his wisdom.
    He would read from his magical book and share stories of adventure.
    The stories were filled with heroes and dragons, magic and mystery.
    Children loved to hear about brave knights and clever princesses.
    The wizard's book never ran out of new tales to tell.
    And so the tradition continued, passed down through generations.
    """ * 20  # Repeat to have more training data
    
    print("Creating tokenizer and model...")
    
    # Initialize tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train([sample_text])  # train expects a list of texts
    
    # Update vocab size based on tokenizer
    config['vocab_size'] = tokenizer.vocab_size
    
    # Create model
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