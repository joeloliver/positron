# Positron

An educational toolkit for training language models from scratch, designed to provide hands-on experience with the complete LLM development pipeline.

## Overview

Positron is a practical learning resource that walks you through every step of building and training language models, from data preparation to inference. The project is structured to help developers understand the fundamentals of modern LLM development without getting lost in production complexity.

## What's Included

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

2. **Run the complete pipeline**
   ```bash
   cd positron-mini
   bash bootstrap_tiny_llm.sh
   ```
   This will:
   - Set up a Python virtual environment
   - Install dependencies (PyTorch, SentencePiece, etc.)
   - Download and prepare the TinyStories dataset
   - Train a tokenizer
   - Pretrain a small GPT model
   - Generate sample text

3. **Try instruction fine-tuning**
   ```bash
   bash sft_quickstart.sh
   ```

4. **Chat with your model**
   ```bash
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
