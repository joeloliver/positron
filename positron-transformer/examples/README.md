# Examples for Pure Python Transformer Implementation

This directory contains practical examples demonstrating how to use the pure Python transformer implementation. Each example is designed to be educational and shows different aspects of transformer functionality.

## 📁 Available Examples

### 🎯 `train_simple.py` - Basic Training Example
**Purpose**: Learn how to train a small transformer model from scratch

**What it demonstrates**:
- Setting up a minimal transformer configuration
- Training on sample text data
- Basic model evaluation
- Saving trained models

**Usage**:
```bash
python examples/train_simple.py
```

**Key Features**:
- Small model (128 embed_dim, 2 layers) for quick training
- Character-level tokenization
- Built-in sample text (fairy tale stories)
- Training progress visualization
- Automatic model checkpointing

**Expected Output**:
```
============================================================
Pure Python Transformer - Simple Training Example
============================================================
Creating tokenizer and model...
Model created with ~65,000 parameters

Starting training...
Epoch 1/5 - Train Loss: 3.2156
Epoch 2/5 - Train Loss: 2.8934
...
Training completed successfully!
```

---

### 🔮 `generate_text.py` - Text Generation Demo
**Purpose**: Generate text using a trained transformer model

**What it demonstrates**:
- Loading saved transformer models
- Autoregressive text generation
- Temperature-controlled sampling
- Interactive text generation

**Usage**:
```bash
# Generate with a prompt
python examples/generate_text.py "Once upon a time"

# Interactive mode
python examples/generate_text.py --interactive

# Custom parameters
python examples/generate_text.py "Hello world" --max-length 50 --temperature 0.8
```

**Command Line Options**:
- `prompt`: Starting text for generation
- `--model-path`: Path to saved model (default: simple_model.npz)
- `--max-length`: Maximum length to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
- `--num-samples`: Number of different samples (default: 1)
- `--interactive`: Start interactive generation session

**Example Output**:
```
Generated text:
"Once upon a time, there was a brave knight who lived in a castle..."
```

---

### 🎯 `visualize_attention.py` - Attention Pattern Visualization
**Purpose**: Understand what the transformer "pays attention" to

**What it demonstrates**:
- Extracting attention weights from trained models
- Visualizing attention patterns as heatmaps
- Understanding attention head behavior
- Analyzing attention statistics

**Usage**:
```bash
# Visualize attention for specific text
python examples/visualize_attention.py "The quick brown fox"

# Compare attention heads
python examples/visualize_attention.py "Sample text" --compare-heads

# Specific layer and head
python examples/visualize_attention.py "Text" --layer 0 --head 2
```

**Features**:
- Heatmap visualization of attention weights
- Token-to-token attention analysis
- Head comparison across same layer
- Statistical analysis of attention patterns
- Export visualizations as PNG files

**Generated Files**:
- `attention_layer_0_head_0.png` - Individual attention heatmaps
- `attention_comparison_layer_0.png` - Multi-head comparison

---

### 🧠 `walkthrough_demo.py` - Complete Process Walkthrough
**Purpose**: Educational deep-dive into every step of transformer processing

**What it demonstrates**:
- Step-by-step transformer pipeline
- Data transformations at each stage
- Detailed mathematical operations
- Parameter analysis and statistics

**Usage**:
```bash
python walkthrough_demo.py
```

**What You'll See**:
1. **Input Preparation**: Text → Characters
2. **Tokenization**: Characters → Token IDs
3. **Embedding**: Token IDs → Dense Vectors
4. **Attention**: Vector interactions and attention weights
5. **Feed-Forward**: Non-linear transformations
6. **Output**: Logits → Probabilities → Next token prediction

**Sample Output**:
```
============================================================
STEP 1: INPUT PREPARATION
============================================================
Input text: 'Once upon a time, there was a wise wizard.'
Text length: 42 characters

============================================================
STEP 2: TOKENIZATION
============================================================
Token IDs: [2, 13, 9, 14, 5, 4, 15, 16, ...]
Number of tokens: 44

📊 Token embeddings:
  Shape: (1, 44, 64)
  Data type: float32
  Min/Max: -1.4654 / 1.6117
  Mean/Std: 0.3942 / 0.6460
```

---

## 🚀 Getting Started

### Prerequisites
```bash
# Ensure you're in the project root
cd positron-transformer

# Activate your virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Quick Start Workflow

1. **First Time Setup**:
```bash
# Train your first model
python examples/train_simple.py
```

2. **Generate Text**:
```bash
# Use the trained model to generate text
python examples/generate_text.py "Your prompt here"
```

3. **Understand the Process**:
```bash
# See detailed walkthrough of what happens
python walkthrough_demo.py
```

4. **Visualize Attention**:
```bash
# See what the model pays attention to
python examples/visualize_attention.py "Your text here"
```

## 🎓 Educational Progression

For learning purposes, we recommend following this order:

1. **`walkthrough_demo.py`** - Understand the theory and process
2. **`train_simple.py`** - Learn practical training
3. **`generate_text.py`** - Experience text generation
4. **`visualize_attention.py`** - Analyze model behavior

## 📊 Model Configurations

All examples use sensible defaults, but you can experiment with different configurations:

### Small Model (Fast Training)
```python
config = {
    'vocab_size': 1000,
    'embed_dim': 64,
    'num_heads': 4,
    'num_layers': 2,
    'ff_dim': 128,
    'max_seq_len': 64
}
# ~65,000 parameters
```

### Medium Model (Better Quality)
```python
config = {
    'vocab_size': 1000,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 4,
    'ff_dim': 512,
    'max_seq_len': 128
}
# ~500,000 parameters
```

### Large Model (Best Quality, Slower)
```python
config = {
    'vocab_size': 1000,
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'ff_dim': 1024,
    'max_seq_len': 256
}
# ~2,000,000 parameters
```

## 🔧 Customization Tips

### Modifying Training Data
```python
# In train_simple.py, replace sample_text with your own data:
with open("your_text_file.txt", "r") as f:
    custom_text = f.read()

# Train on your custom data
train_model(model, tokenizer, custom_text, train_config)
```

### Adjusting Generation Parameters
```python
# Higher temperature = more creative/random
generated = model.generate(prompt, tokenizer, temperature=1.2)

# Lower temperature = more conservative/predictable  
generated = model.generate(prompt, tokenizer, temperature=0.3)
```

### Custom Tokenization
```python
# Use different tokenizer types
tokenizer = WordTokenizer()  # Word-level
tokenizer = SimpleBPETokenizer()  # Byte-pair encoding
```

## 📈 Performance Expectations

### Training Speed (per epoch)
- **Small Model**: ~10-30 seconds
- **Medium Model**: ~1-3 minutes  
- **Large Model**: ~5-15 minutes

*Note: Pure Python implementation is educational - expect slower training than PyTorch/TensorFlow*

### Memory Usage
- **Small Model**: ~50MB RAM
- **Medium Model**: ~200MB RAM
- **Large Model**: ~500MB RAM

### Text Quality
- **After 5 epochs**: Basic character patterns
- **After 20 epochs**: Simple word formation
- **After 50+ epochs**: Coherent short phrases

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'config_py'**
```bash
# Make sure you're running from the project root
cd positron-transformer
python examples/train_simple.py
```

**CUDA/GPU Errors**
```bash
# This implementation uses CPU only (NumPy)
# No GPU setup required
```

**Memory Errors**
```python
# Reduce model size or batch size in config
config['embed_dim'] = 32  # Smaller embedding
train_config['batch_size'] = 4  # Smaller batches
```

**Poor Text Quality**
```python
# Try these improvements:
config['num_layers'] = 6  # More layers
config['embed_dim'] = 256  # Larger embeddings  
train_config['num_epochs'] = 100  # More training
```

## 📚 Additional Resources

### Understanding Transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) - Mathematical details

### Code Architecture
- `../config_py.py` - All configuration options
- `../transformer.py` - Core model implementation  
- `../training.py` - Training loops and optimization
- `../attention_py.py` - Multi-head attention mechanism

### Extending the Implementation
1. **Add new optimizers** in `training.py`
2. **Implement new attention variants** in `attention_py.py`
3. **Create custom tokenizers** in `tokenizer_py.py`
4. **Add model architectures** in `transformer.py`

---

## 🎯 Next Steps

Once you're comfortable with these examples:

1. **Experiment with different text domains** (code, poetry, dialogue)
2. **Try different model architectures** (more layers, heads, dimensions)  
3. **Implement advanced features** (beam search, top-k sampling)
4. **Scale to larger datasets** (books, articles, code repositories)
5. **Explore attention patterns** in different domains

## 💡 Tips for Success

- **Start small**: Use the default configurations first
- **Monitor training**: Watch loss curves and sample outputs
- **Experiment gradually**: Change one parameter at a time
- **Save checkpoints**: Don't lose training progress
- **Visualize results**: Use attention maps to understand behavior

Remember: This is an educational implementation designed for understanding transformers from first principles. The code prioritizes clarity and mathematical transparency over speed!

---

*Happy learning! 🚀*