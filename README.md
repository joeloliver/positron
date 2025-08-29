# Positron

An educational toolkit for training language models from scratch, designed to provide hands-on experience with the complete LLM development pipeline.

## Overview

Positron is a practical learning resource that walks you through every step of building and training language models, from data preparation to inference. The project is structured to help developers understand the fundamentals of modern LLM development without getting lost in production complexity.

## What's Included

### 🚀 Positron-Mini

A complete, highly-commented mini framework located in `positron-mini/` that demonstrates:

- **End-to-end pretraining pipeline** (`bootstrap_tiny_llm.sh`)
  - Virtual environment setup and dependency management
  - Dataset preparation using TinyStories corpus
  - SentencePiece tokenizer training (16k BPE vocabulary)
  - TinyGPT model pretraining (~15M parameters)
  - Text generation and sampling

- **Supervised Fine-Tuning (SFT)** (`sft_quickstart.sh`)
  - Instruction fine-tuning from pretrained checkpoints
  - Example instruction dataset for validation
  - Alignment training for better instruction following

- **Sample datasets and configs**
  - `tiny_instruct_sample.jsonl` - Toy instruction dataset
  - Detailed documentation explaining each component

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

## Key Features

- **🎓 Educational Focus**: Extensively commented code explaining the "why" behind each step
- **⚡ Fast Iteration**: Uses tiny datasets and models for quick experimentation
- **🔧 Complete Pipeline**: From raw text to instruction-following model
- **🏗️ Modular Design**: Each component can be understood and modified independently
- **📚 Self-Contained**: Minimal dependencies and clear setup instructions

## Requirements

- Python 3.7+
- 2-4 GB RAM (for tiny models)
- Optional: CUDA-compatible GPU (scripts work on CPU too)

## Educational Value

This toolkit is perfect for:
- Understanding transformer pretraining mechanics
- Learning tokenization and data preprocessing
- Experimenting with model architectures
- Exploring instruction fine-tuning techniques
- Building intuition about training dynamics

## Notes

- **Fast Development**: Uses TinyStories dataset and small models for rapid iteration
- **Scalability**: Easily adaptable to larger datasets and models
- **Platform Support**: Works on CPU, CUDA GPUs, and Apple Silicon (MPS)
- **Flexibility**: All scripts can be customized for different experiments

For detailed explanations of each component, see [`positron-mini/README_QUICKSTART.md`](positron-mini/README_QUICKSTART.md).

## License

See [LICENSE](LICENSE) for details.
