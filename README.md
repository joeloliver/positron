# Positron

An educational journey through language model development, evolving from FPGA-based neural networks to modern transformer architectures. This project bridges hardware and software understanding of machine learning systems.

## Overview

Positron is a practical learning resource that walks you through every step of building and training language models, from data preparation to inference. The project is structured to help developers understand the fundamentals of modern LLM development without getting lost in production complexity.

## What's Included

### 🧠 Positron-Transformer

A complete transformer implementation in pure Python/NumPy that represents a personal learning journey from FPGA neural networks to modern language models:

- **🐍 Pure Python Implementation** - Mathematical transparency from the ground up
  - 100% NumPy-based numerical computation, no frameworks
  - Every component built from scratch with full forward/backward passes
  - Drawing from hardware neural network experience for numerical stability
  - Educational focus bridging VHDL/Verilog hardware design to Python mathematics

- **🏗️ Complete Architecture** - Evolution from MLPs to attention mechanisms
  - Multi-head self-attention (the leap from fixed weights to dynamic attention)
  - Feed-forward networks with GELU/ReLU/Swish activations
  - Layer normalization and residual connections
  - Token and positional embeddings
  - Multiple tokenizer options (character, word, BPE)

- **🎯 Working Training & Inference** - From 160MHz FPGA constraints to software flexibility
  - Adam optimizer implementation applying IEEE 754 precision principles
  - Autoregressive text generation with temperature sampling
  - Centralized configuration system for reproducible experiments
  - Training loss visualization and comprehensive metrics
  - 20-200 epoch training with real-time progress monitoring

- **🔬 Educational Value** - Understanding transformers at the mathematical level
  - Complete derivations for all operations
  - Hardware-aware design principles from FPGA background
  - Bridge from sequential processing to parallel attention heads
  - Foundation for potential future FPGA acceleration

### 🚀 Positron-Mini

A complete, highly-optimized mini framework located in `positron-mini/` that demonstrates:

- **🍎 Apple Silicon Optimized Training** (`bootstrap_tiny_llm.sh`)
  - M3 Max / Apple Silicon auto-detection and optimization
  - bfloat16 precision with MPS backend for 2-3x speedup
  - Optimized batch sizes for unified memory architecture
  - Virtual environment setup and dependency management
  - Dataset preparation using TinyStories corpus
  - SentencePiece tokenizer training (16k BPE vocabulary)
  - TinyGPT model pretraining (~15M parameters, 5000 steps)
  - Real-time training progress monitoring

- **Supervised Fine-Tuning (SFT)** (`sft_quickstart.sh`)
  - Instruction fine-tuning from pretrained checkpoints
  - Example instruction dataset for validation
  - Alignment training for better instruction following

- **⚡ Incremental Training** (`train_incremental.sh`)
  - Train in configurable chunks (e.g., 1000 steps at a time)
  - Test model quality at each milestone before continuing
  - Automatic resume from checkpoints - never lose progress
  - Stop early if results are good enough

- **💬 Interactive Chat Interface** (`ask_model.sh`)
  - Real-time question answering with your trained model
  - Interactive and single-question modes
  - Configurable generation parameters (temperature, top_p, max_tokens)
  - Live settings adjustment during chat sessions

- **Sample datasets and configs**
  - `tiny_instruct_sample.jsonl` - Toy instruction dataset
  - Comprehensive documentation explaining each component

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd positron
   ```

2. **Start with the pure Python transformer** (recommended for learning)

   ```bash
   cd positron-transformer

   # Install minimal dependencies (just NumPy and matplotlib)
   pip install numpy matplotlib

   # Train a transformer from scratch (~2-3 minutes for quick test)
   python examples/train_simple.py

   # Or train longer for better results (~10-15 minutes)
   python examples/train_simple.py --long-training

   # Generate text with your trained model
   python examples/generate_text.py "Once upon a time"
   ```

   **Or explore the PyTorch-based framework:**

   ```bash
   cd positron-mini

   # Full training (5000 steps at once)
   bash bootstrap_tiny_llm.sh

   # Or incremental training (recommended)
   bash train_incremental.sh 1000  # Train in 1000-step chunks
   ```

3. **Try instruction fine-tuning (positron-mini only)**
   ```bash
   cd positron-mini
   bash sft_quickstart.sh
   ```

4. **Chat with your model (positron-mini only)**
   ```bash
   cd positron-mini
   # Interactive mode
   bash ask_model.sh

   # Single question
   bash ask_model.sh "Explain how neural networks work"
   ```

## Key Features

- **🎓 Educational Focus**: Extensively commented code explaining the "why" behind each step
- **🍎 Apple Silicon First**: Optimized for M-series chips with automatic hardware detection
- **⚡ Fast Iteration**: Uses tiny datasets and models for quick experimentation  
- **🔧 Complete Pipeline**: From raw text to interactive chat model
- **🏗️ Modular Design**: Each component can be understood and modified independently
- **📚 Self-Contained**: Minimal dependencies and clear setup instructions
- **💬 Interactive Interface**: Chat with your trained models via simple shell script

## Requirements

- Python 3.9+
- 8-16 GB RAM (recommended for optimal training)
- **Recommended**: Apple Silicon (M1/M2/M3) for best performance
- **Alternative**: CUDA-compatible GPU or CPU (automatic fallback)

## Educational Value

This toolkit is perfect for:
- Understanding transformer pretraining mechanics
- Learning tokenization and data preprocessing
- Experimenting with model architectures
- Exploring instruction fine-tuning techniques
- Building intuition about training dynamics

## Performance Optimizations

| Hardware | Precision | Batch Tokens | Backend | Typical Speed |
|----------|-----------|--------------|---------|---------------|
| **M3 Max** | bfloat16 | 98,304 | MPS | ~7000 tok/s |
| M1/M2 | bfloat16 | 98,304 | MPS | ~5000 tok/s |
| CUDA GPU | bfloat16 | 131,072 | CUDA | ~8000+ tok/s |
| CPU | float32 | 65,536 | CPU | ~1000 tok/s |

## Notes

- **Apple Silicon Optimized**: Automatically detects and optimizes for M-series chips
- **Fast Development**: Uses TinyStories dataset and small models for rapid iteration  
- **Scalability**: Easily adaptable to larger datasets and models
- **Platform Support**: Works on CPU, CUDA GPUs, and Apple Silicon (MPS)
- **Flexibility**: All scripts can be customized for different experiments
- **Real-time Monitoring**: Training progress shown for every step

For detailed setup instructions, usage examples, and troubleshooting, see [`positron-mini/README.md`](positron-mini/README.md).

## License

See [LICENSE](LICENSE) for details.
