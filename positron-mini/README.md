# 🚀 Positron Mini

A minimal, efficient GPT implementation optimized for Apple Silicon (M-series chips) and designed for educational exploration of language model training and fine-tuning.

## ✨ Features

- **🍎 Apple Silicon Optimized**: Full M3 Max support with bfloat16 precision and MPS backend
- **⚡ Efficient Training**: Optimized batch sizes and memory usage for unified memory architecture  
- **🎯 Simple Fine-tuning**: Easy supervised fine-tuning (SFT) pipeline
- **💬 Interactive Chat**: Built-in question-answering interface via `ask_model.sh`
- **📚 TinyStories Dataset**: Pre-configured for coherent story generation training

## 🏗️ Architecture

- **Model**: 15M parameter GPT-style transformer
- **Context Length**: 512 tokens
- **Vocabulary**: 16,000 tokens (SentencePiece BPE)
- **Layers**: 8 transformer blocks with 8 attention heads

## 🚀 Quick Start

### 1. Setup & Base Model Training
```bash
# Clone and setup
git clone <your-repo>
cd positron-mini

# Train base model (optimized for M3 Max)
./bootstrap_tiny_llm.sh
```

This will:
- Create virtual environment with optimized PyTorch
- Download and prepare TinyStories dataset  
- Train for 5000 steps with Apple Silicon optimizations
- Generate a sample to test the base model

### 2. Supervised Fine-Tuning (Optional)
```bash
# Fine-tune for instruction following
./sft_quickstart.sh
```

### 3. Interactive Question Answering
```bash
# Ask questions to your model
./ask_model.sh

# Or single question mode
./ask_model.sh "Tell me about gravity"
```

## 🛠️ Manual Usage

### Training from Scratch
```bash
source .venv/bin/activate

python train.py \
  --config TinyGPT \
  --dataset_path data/tinystories.txt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --context_len 512 \
  --batch_tokens 98304 \
  --max_steps 5000 \
  --warmup_steps 500 \
  --lr 3e-4 \
  --precision bf16 \
  --out_dir runs/my-model
```

### Text Generation
```bash
python generate.py \
  --ckpt runs/positron-mini-15M-v1-base/latest.pt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --prompt "Once upon a time" \
  --max_tokens 80 \
  --temperature 0.7
```

### Fine-tuning 
```bash
python sft_train.py \
  --base_ckpt runs/positron-mini-15M-v1-base/latest.pt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --sft_data tiny_instruct_sample.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --lr 1e-5 \
  --out_dir runs/my-sft-model
```

## 🍎 Apple Silicon Optimizations

The project automatically detects and optimizes for Apple Silicon:

| Device | Precision | Batch Tokens | Backend |
|--------|-----------|--------------|---------|
| M3 Max | bfloat16  | 98,304       | MPS     |
| CUDA   | bfloat16  | 131,072      | CUDA    |
| CPU    | float32   | 65,536       | CPU     |

**M3 Max Benefits:**
- ⚡ 2-3x faster training with bfloat16
- 💾 Optimized memory usage for unified memory
- 🔥 Metal Performance Shaders acceleration

## 📁 Project Structure

```
positron-mini/
├── 🤖 Core Training
│   ├── train.py              # Main training script
│   ├── gpt_min.py            # GPT model implementation  
│   └── generate.py           # Text generation
├── 🎯 Fine-tuning
│   ├── sft_train.py          # Supervised fine-tuning
│   └── tiny_instruct_sample.jsonl  # Sample instruction data
├── 🚀 Scripts  
│   ├── bootstrap_tiny_llm.sh # Complete setup + base training
│   ├── sft_quickstart.sh     # Quick fine-tuning
│   └── ask_model.sh          # Interactive chat interface
├── 📊 Data & Models
│   ├── data/                 # Training datasets
│   ├── tokenizer/            # SentencePiece tokenizer
│   └── runs/                 # Model checkpoints
└── 📝 Documentation
    ├── README.md             # This file
    └── README_QUICKSTART.md  # Quick reference
```

## 🎮 Interactive Features

### `ask_model.sh` - Chat Interface

**Interactive Mode:**
```bash
./ask_model.sh

🚀 Positron Mini Interactive Question Mode
❓ Your question: Explain gravity to a child.
🤖 Generating response...

# Type 'settings' to adjust parameters
# Type 'quit' to exit
```

**Single Question:**
```bash
./ask_model.sh "Write a short story about a robot"
```

**Configurable Parameters:**
- `max_tokens`: Response length (default: 30)
- `temperature`: Creativity (default: 0.3) 
- `top_p`: Nucleus sampling (default: 0.8)

## 📋 Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+ with MPS support
- **Hardware**: Apple Silicon recommended (works on any device)
- **Memory**: 8GB+ RAM recommended for training
- **Storage**: 2GB+ for datasets and models

## 🔧 Development

### Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch sentencepiece datasets tqdm
```

### Training Tips

**For Better Results:**
- Use more training steps (5000+ recommended)
- Experiment with learning rates (1e-4 to 5e-4)
- Try different temperature values for generation (0.1-1.0)

**For Faster Training:**
- Use bfloat16 precision on supported hardware
- Increase batch size if you have more memory
- Monitor GPU/MPS utilization

## 🐛 Troubleshooting

**Training Stuck?**
- Check if MPS is enabled: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Monitor with: `ps aux | grep python`
- Reduce batch size if OOM errors

**Poor Generation Quality?**
- Ensure base model trained for 3000+ steps
- Try lower temperature (0.1-0.5) for more coherent text
- Check if SFT model exists and is properly trained

**Memory Issues?**
- Reduce `batch_tokens` in bootstrap script
- Use fp32 instead of bf16 if needed
- Monitor memory with Activity Monitor

## 📊 Training Metrics

**Typical Training Progress:**
```
[step 1/5000]    loss=9.756  lr=6.00e-07  ~7000 tok/s
[step 500/5000]  loss=6.234  lr=3.00e-04  ~7200 tok/s  
[step 2500/5000] loss=3.456  lr=1.50e-04  ~7100 tok/s
[step 5000/5000] loss=2.123  lr=0.00e+00  ~7000 tok/s
```

**Good Loss Targets:**
- **Base Training**: Start ~10.0, end ~2.0-3.0
- **Fine-tuning**: Start ~3.0, end ~1.5-2.5

## 🤝 Contributing

This is an educational project focused on understanding transformer training and Apple Silicon optimization. Feel free to:

- Experiment with different architectures
- Try new datasets beyond TinyStories  
- Optimize further for different hardware
- Add new generation techniques

## 📜 License

MIT License - feel free to use for learning and experimentation!

---

**Happy Training! 🎉**

*Optimized for Apple Silicon by design, educational by nature.*
