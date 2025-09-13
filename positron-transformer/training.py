"""
Training Implementation for Pure Python Transformer

This module implements the complete training loop with optimizers,
loss functions, and training utilities. Based on decades of neural
network implementation experience.

Author: Joel Oliver
Based on: Pure Python Transformer Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Callable
import os
import time
from config_py import MODEL_CONFIG, TRAINING_CONFIG
from transformer import Transformer
from tokenizer_py import SimpleTokenizer
from utils_py import compute_cross_entropy_loss, clip_gradients


class AdamOptimizer:
    """
    Adam optimizer implementation from scratch
    
    Implements the Adam optimization algorithm with bias correction
    and numerical stability improvements.
    """
    
    def __init__(self, learning_rate: float = 1e-3, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # State variables
        self.t = 0  # Time step
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def zero_grad(self) -> None:
        """Reset gradients (conceptually - handled by parameter updates)"""
        pass
    
    def step(self, parameters: Dict[str, np.ndarray], 
             gradients: Dict[str, np.ndarray]) -> None:
        """
        Perform one optimization step
        
        Args:
            parameters: Dictionary of parameter arrays
            gradients: Dictionary of gradient arrays (same keys as parameters)
        """
        self.t += 1
        
        for name, param in parameters.items():
            if name not in gradients:
                continue
            
            grad = gradients[name]
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize moment estimates if needed
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # Update biased first and second moment estimates
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LearningRateScheduler:
    """Learning rate scheduling utilities"""
    
    @staticmethod
    def linear_warmup_cosine_decay(step: int, warmup_steps: int, 
                                 total_steps: int, base_lr: float, 
                                 min_lr: float = 0.0) -> float:
        """
        Linear warmup followed by cosine decay
        
        Args:
            step: Current training step
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            base_lr: Peak learning rate
            min_lr: Minimum learning rate
        
        Returns:
            Current learning rate
        """
        if step < warmup_steps:
            # Linear warmup
            return base_lr * step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


class Trainer:
    """
    Complete training implementation for the transformer model
    """
    
    def __init__(self, model: Transformer, tokenizer: SimpleTokenizer, 
                 config: Dict[str, Any] = None):
        if config is None:
            config = TRAINING_CONFIG
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(
            learning_rate=config['learning_rate'],
            beta1=config.get('adam_beta1', 0.9),
            beta2=config.get('adam_beta2', 0.999),
            epsilon=config.get('adam_epsilon', 1e-8),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_step(self, batch: Dict[str, np.ndarray]) -> float:
        """
        Perform one training step
        
        Args:
            batch: Dictionary containing 'input_ids' and 'labels'
            
        Returns:
            Loss value for this step
        """
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # Forward pass
        logits = self.model.forward(input_ids)
        
        # Compute loss
        loss = compute_cross_entropy_loss(logits, labels)
        
        # Backward pass
        grad_logits = self._compute_loss_gradient(logits, labels)
        self.model.backward(grad_logits)
        
        # Collect all parameters and gradients
        parameters, gradients = self._collect_parameters_and_gradients()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            gradients = clip_gradients(gradients, self.config['grad_clip'])
        
        # Optimizer step
        self.optimizer.step(parameters, gradients)
        
        # Update learning rate
        if self.config.get('use_lr_schedule', True):
            new_lr = LearningRateScheduler.linear_warmup_cosine_decay(
                self.step,
                self.config.get('warmup_steps', 1000),
                self.config.get('total_steps', 100000),
                self.config['learning_rate']
            )
            self.optimizer.lr = new_lr
            self.learning_rates.append(new_lr)
        
        self.step += 1
        return loss
    
    def evaluate(self, val_data: List[Dict[str, np.ndarray]]) -> float:
        """
        Evaluate model on validation data
        
        Args:
            val_data: List of validation batches
            
        Returns:
            Average validation loss
        """
        self.model.training = False
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_data:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass only
            logits = self.model.forward(input_ids)
            loss = compute_cross_entropy_loss(logits, labels)
            
            total_loss += loss
            num_batches += 1
        
        self.model.training = True
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self, train_data: List[Dict[str, np.ndarray]], 
              val_data: Optional[List[Dict[str, np.ndarray]]] = None,
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            train_data: List of training batches
            val_data: Optional validation data
            num_epochs: Number of epochs to train (overrides config)
            
        Returns:
            Dictionary with training metrics
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {self.model.get_num_parameters():,} parameters")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training loop
            for batch_idx, batch in enumerate(train_data):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Log progress
                if (batch_idx + 1) % self.config.get('log_interval', 100) == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.get('log_interval', 100):])
                    lr = self.optimizer.lr if hasattr(self.optimizer, 'lr') else self.config['learning_rate']
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_data)}, "
                          f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")
            
            # Epoch metrics
            epoch_train_loss = np.mean(epoch_losses)
            self.train_losses.append(epoch_train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.evaluate(val_data)
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model.npz')
            else:
                print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.npz')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint"""
        self.model.save_checkpoint(filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint"""
        self.model.load_checkpoint(filepath)
        print(f"Checkpoint loaded from {filepath}")
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Training Loss', alpha=0.8)
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Validation Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        if self.learning_rates:
            axes[1].plot(self.learning_rates, label='Learning Rate', alpha=0.8)
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def _compute_loss_gradient(self, logits: np.ndarray, 
                             labels: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss w.r.t. logits
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            labels: True labels of shape (batch_size, seq_len)
            
        Returns:
            Gradient w.r.t. logits
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for easier processing
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        
        # Compute softmax probabilities
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # One-hot encode labels
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels_flat)), labels_flat] = 1
        
        # Gradient of cross-entropy loss
        grad_flat = (probs - one_hot) / batch_size
        
        # Reshape back
        grad_logits = grad_flat.reshape(batch_size, seq_len, vocab_size)
        
        return grad_logits
    
    def _collect_parameters_and_gradients(self) -> Tuple[Dict[str, np.ndarray], 
                                                        Dict[str, np.ndarray]]:
        """
        Collect all model parameters and their gradients
        
        Returns:
            Tuple of (parameters_dict, gradients_dict)
        """
        parameters = {}
        gradients = {}
        
        # Collect parameters and gradients from model components
        # This is a simplified version - in practice, you'd need to implement
        # parameter collection methods in each component
        
        # Output projection
        parameters['output_projection'] = self.model.output_projection
        gradients['output_projection'] = getattr(self.model, 'grad_output_projection', 
                                                np.zeros_like(self.model.output_projection))
        
        if self.model.output_bias is not None:
            parameters['output_bias'] = self.model.output_bias
            gradients['output_bias'] = getattr(self.model, 'grad_output_bias',
                                             np.zeros_like(self.model.output_bias))
        
        return parameters, gradients


def create_training_data(text: str, tokenizer: SimpleTokenizer, 
                        seq_len: int, batch_size: int) -> List[Dict[str, np.ndarray]]:
    """
    Create training data from text
    
    Args:
        text: Input text string
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        batch_size: Batch size
        
    Returns:
        List of training batches
    """
    # Tokenize text
    tokens = tokenizer.encode(text)
    
    # Create sequences
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        input_seq = tokens[i:i + seq_len]
        target_seq = tokens[i + 1:i + seq_len + 1]  # Shifted by 1 for next-token prediction
        sequences.append((input_seq, target_seq))
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        # Pad batch if necessary
        while len(batch_sequences) < batch_size and len(batch_sequences) > 0:
            batch_sequences.append(batch_sequences[-1])  # Repeat last sequence
        
        if len(batch_sequences) == batch_size:
            input_ids = np.array([seq[0] for seq in batch_sequences])
            labels = np.array([seq[1] for seq in batch_sequences])
            
            batches.append({
                'input_ids': input_ids,
                'labels': labels
            })
    
    return batches


def train_model(model: Transformer, tokenizer: SimpleTokenizer, 
               text_data: str, config: Dict[str, Any] = None) -> Dict[str, List[float]]:
    """
    Convenience function to train a model on text data
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer instance
        text_data: Training text
        config: Training configuration
        
    Returns:
        Training metrics
    """
    if config is None:
        config = TRAINING_CONFIG
    
    # Create training data
    train_data = create_training_data(
        text_data, tokenizer, 
        seq_len=config.get('seq_len', 128),
        batch_size=config.get('batch_size', 32)
    )
    
    # Create trainer and train
    trainer = Trainer(model, tokenizer, config)
    metrics = trainer.train(train_data)
    
    return metrics


if __name__ == "__main__":
    # Simple training test
    from config_py import MODEL_CONFIG
    
    # Create small model for testing
    config = MODEL_CONFIG.copy()
    config['vocab_size'] = 1000
    config['embed_dim'] = 128
    config['num_heads'] = 4
    config['num_layers'] = 2
    
    model = Transformer(config)
    tokenizer = SimpleTokenizer()
    
    # Test training on dummy data
    dummy_text = "This is a test. " * 1000
    
    print("Testing training loop...")
    metrics = train_model(model, tokenizer, dummy_text)
    print("Training test completed!")