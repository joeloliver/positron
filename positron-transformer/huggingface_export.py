"""
Hugging Face Compatible Model Export/Import
Pure Python implementation for exporting models in HF-compatible format
"""

import json
import numpy as np
import struct
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os

class HuggingFaceExporter:
    """Export models in Hugging Face compatible format"""
    
    # Standard HF naming conventions for transformer weights
    HF_KEY_MAPPING = {
        # Embeddings
        'embedding.token_embedding.weight': 'embeddings.word_embeddings.weight',
        'embedding.pos_encoding.weight': 'embeddings.position_embeddings.weight',
        'embedding.norm.weight': 'embeddings.LayerNorm.weight',
        'embedding.norm.bias': 'embeddings.LayerNorm.bias',
        
        # Output projection
        'output_projection': 'lm_head.weight',
        'output_bias': 'lm_head.bias',
        
        # Final layer norm
        'final_norm.weight': 'transformer.ln_f.weight',
        'final_norm.bias': 'transformer.ln_f.bias',
    }
    
    @staticmethod
    def get_hf_key(key: str) -> str:
        """Convert internal key to HF-compatible key"""
        # Direct mapping
        if key in HuggingFaceExporter.HF_KEY_MAPPING:
            return HuggingFaceExporter.HF_KEY_MAPPING[key]
        
        # Layer-specific mappings
        if key.startswith('layer_'):
            # Extract layer number and component
            parts = key.split('.')
            layer_num = int(parts[0].split('_')[1])
            component = '.'.join(parts[1:])
            
            # Map component names
            component_mapping = {
                'attention.q_proj.weight': f'transformer.h.{layer_num}.attn.q_proj.weight',
                'attention.q_proj.bias': f'transformer.h.{layer_num}.attn.q_proj.bias',
                'attention.k_proj.weight': f'transformer.h.{layer_num}.attn.k_proj.weight',
                'attention.k_proj.bias': f'transformer.h.{layer_num}.attn.k_proj.bias',
                'attention.v_proj.weight': f'transformer.h.{layer_num}.attn.v_proj.weight',
                'attention.v_proj.bias': f'transformer.h.{layer_num}.attn.v_proj.bias',
                'attention.out_proj.weight': f'transformer.h.{layer_num}.attn.out_proj.weight',
                'attention.out_proj.bias': f'transformer.h.{layer_num}.attn.out_proj.bias',
                
                'feed_forward.linear1.weight': f'transformer.h.{layer_num}.mlp.fc_in.weight',
                'feed_forward.linear1.bias': f'transformer.h.{layer_num}.mlp.fc_in.bias',
                'feed_forward.linear2.weight': f'transformer.h.{layer_num}.mlp.fc_out.weight',
                'feed_forward.linear2.bias': f'transformer.h.{layer_num}.mlp.fc_out.bias',
                
                'norm1.weight': f'transformer.h.{layer_num}.ln_1.weight',
                'norm1.bias': f'transformer.h.{layer_num}.ln_1.bias',
                'norm2.weight': f'transformer.h.{layer_num}.ln_2.weight',
                'norm2.bias': f'transformer.h.{layer_num}.ln_2.bias',
            }
            
            if component in component_mapping:
                return component_mapping[component]
        
        # Return original if no mapping found
        return key
    
    @staticmethod
    def export_config(model_config: Dict[str, Any], output_dir: str) -> None:
        """Export model configuration in HF format"""
        # Map our config to standard HF config
        hf_config = {
            "model_type": "gpt2",  # Using GPT-2 as reference architecture
            "vocab_size": model_config.get("vocab_size", 50257),
            "n_positions": model_config.get("max_seq_len", 1024),
            "n_embd": model_config.get("embed_dim", 768),
            "n_layer": model_config.get("num_layers", 12),
            "n_head": model_config.get("num_heads", 12),
            "n_inner": model_config.get("ff_dim", None),
            "activation_function": model_config.get("activation", "gelu"),
            "resid_pdrop": model_config.get("dropout", 0.1),
            "embd_pdrop": model_config.get("dropout", 0.1),
            "attn_pdrop": model_config.get("attention_dropout", 0.1),
            "layer_norm_epsilon": model_config.get("layer_norm_eps", 1e-5),
            "initializer_range": model_config.get("init_std", 0.02),
            "use_cache": True,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 0,
            "architectures": ["GPT2LMHeadModel"],
            "task_specific_params": {
                "text-generation": {
                    "do_sample": True,
                    "max_length": 50
                }
            }
        }
        
        # If ff_dim is not specified, use 4 * n_embd (standard for GPT-2)
        if hf_config["n_inner"] is None:
            hf_config["n_inner"] = 4 * hf_config["n_embd"]
        
        # Save config
        config_path = Path(output_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(hf_config, f, indent=2)
        
        print(f"Config saved to {config_path}")
    
    @staticmethod
    def export_weights_safetensors(weights: Dict[str, np.ndarray], output_dir: str) -> None:
        """
        Export weights in SafeTensors format (pure Python implementation)
        SafeTensors format: https://github.com/huggingface/safetensors
        """
        output_path = Path(output_dir) / "model.safetensors"
        
        # Convert weights to HF naming
        hf_weights = {}
        for key, value in weights.items():
            hf_key = HuggingFaceExporter.get_hf_key(key)
            # Ensure weights are float32
            hf_weights[hf_key] = value.astype(np.float32)
        
        # Create SafeTensors file
        metadata = {}
        tensor_info = {}
        data_offsets = []
        current_offset = 0
        
        # Build metadata for each tensor
        for name, tensor in hf_weights.items():
            shape = list(tensor.shape)
            dtype = "F32"  # We're using float32
            data_size = tensor.nbytes
            
            tensor_info[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [current_offset, current_offset + data_size]
            }
            
            data_offsets.append((name, current_offset, tensor))
            current_offset += data_size
        
        # Create header
        header = {
            "__metadata__": metadata,
            **tensor_info
        }
        
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')
        header_size = len(header_bytes)
        
        # Align header to 8 bytes
        padding_size = (8 - header_size % 8) % 8
        header_bytes += b' ' * padding_size
        
        # Write file
        with open(output_path, 'wb') as f:
            # Write header size (8 bytes, little-endian)
            f.write(struct.pack('<Q', len(header_bytes)))
            # Write header
            f.write(header_bytes)
            # Write tensor data
            for name, offset, tensor in data_offsets:
                f.write(tensor.tobytes())
        
        print(f"Weights saved in SafeTensors format to {output_path}")
    
    @staticmethod
    def export_weights_numpy(weights: Dict[str, np.ndarray], output_dir: str) -> None:
        """Export weights in numpy format with HF-compatible naming"""
        output_path = Path(output_dir) / "pytorch_model.bin"
        
        # Convert to HF naming
        hf_weights = {}
        for key, value in weights.items():
            hf_key = HuggingFaceExporter.get_hf_key(key)
            hf_weights[hf_key] = value
        
        # Save as numpy archive
        np.savez_compressed(output_path, **hf_weights)
        print(f"Weights saved in numpy format to {output_path}")
    
    @staticmethod
    def create_model_card(model_config: Dict[str, Any], output_dir: str, 
                         model_name: str = "pure-python-transformer") -> None:
        """Create a model card (README.md) for Hugging Face"""
        model_card = f"""---
license: apache-2.0
tags:
- pure-python
- transformer
- text-generation
language:
- en
library_name: transformers
---

# {model_name}

This model was trained using a pure Python transformer implementation (no PyTorch/TensorFlow).

## Model Details

- **Model Type**: GPT-2 style autoregressive transformer
- **Embedding Dimension**: {model_config.get('embed_dim', 768)}
- **Number of Layers**: {model_config.get('num_layers', 12)}
- **Number of Heads**: {model_config.get('num_heads', 12)}
- **Vocabulary Size**: {model_config.get('vocab_size', 50257)}
- **Context Length**: {model_config.get('max_seq_len', 1024)}
- **Feed-Forward Dimension**: {model_config.get('ff_dim', 'auto')}

## Training

This model was trained using a custom pure Python implementation without any deep learning frameworks.

## Usage

### With Transformers Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### With Pure Python Implementation

```python
from transformer import Transformer
from tokenizer_py import CharacterTokenizer

model = Transformer(config)
model.load_checkpoint("model.npz")

tokenizer = CharacterTokenizer()
tokenizer.load_vocab("tokenizer.vocab")

generated = model.generate("Once upon a time", tokenizer, max_length=50)
```

## Limitations

- This is a small model trained on limited data
- Primarily for educational and research purposes
- May not perform as well as larger, framework-optimized models

## Technical Details

The model uses:
- Layer normalization
- Multi-head self-attention
- GELU activation function
- Residual connections
- Positional embeddings (sinusoidal or learned)
"""
        
        readme_path = Path(output_dir) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        print(f"Model card saved to {readme_path}")
    
    @staticmethod
    def export_model(model, tokenizer, output_dir: str, model_name: str = "pure-python-transformer") -> None:
        """Complete export pipeline for Hugging Face compatibility"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting model to {output_dir}...")
        
        # 1. Export configuration
        HuggingFaceExporter.export_config(model.config, output_dir)
        
        # 2. Get all weights
        checkpoint = {}
        checkpoint['output_projection'] = model.output_projection
        if model.output_bias is not None:
            checkpoint['output_bias'] = model.output_bias
        
        # Embedding weights
        emb_params = model.embedding.get_parameters()
        for key, value in emb_params.items():
            checkpoint[f'embedding.{key}'] = value
        
        # Final layer norm
        final_norm_params = model.final_norm.get_parameters()
        for key, value in final_norm_params.items():
            checkpoint[f'final_norm.{key}'] = value
        
        # Transformer layers
        for i, layer in enumerate(model.layers):
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                checkpoint[f'layer_{i}.{key}'] = value
        
        # 3. Export weights in SafeTensors format
        HuggingFaceExporter.export_weights_safetensors(checkpoint, output_dir)
        
        # 4. Also export in numpy format as backup
        HuggingFaceExporter.export_weights_numpy(checkpoint, output_dir)
        
        # 5. Export tokenizer vocab
        if tokenizer is not None:
            tokenizer_path = output_path / "tokenizer.json"
            # Export tokenizer config in HF format
            tokenizer_config = {
                "model_type": "gpt2",
                "vocab_size": tokenizer.vocab_size,
                "pad_token": tokenizer.pad_token,
                "unk_token": tokenizer.unk_token,
                "bos_token": tokenizer.bos_token,
                "eos_token": tokenizer.eos_token,
            }
            with open(output_path / "tokenizer_config.json", 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            
            # Save vocabulary
            tokenizer.save_vocab(str(output_path / "vocab.txt"))
        
        # 6. Create model card
        HuggingFaceExporter.create_model_card(model.config, output_dir, model_name)
        
        print(f"\n✅ Model exported successfully to {output_dir}")
        print("Files created:")
        print("  - config.json (model configuration)")
        print("  - model.safetensors (weights in SafeTensors format)")
        print("  - pytorch_model.bin.npz (weights in numpy format)")
        print("  - README.md (model card)")
        if tokenizer is not None:
            print("  - tokenizer_config.json (tokenizer configuration)")
            print("  - vocab.txt (vocabulary)")


class HuggingFaceImporter:
    """Import models from Hugging Face compatible format"""
    
    @staticmethod
    def load_safetensors(filepath: str) -> Dict[str, np.ndarray]:
        """Load weights from SafeTensors format"""
        weights = {}
        
        with open(filepath, 'rb') as f:
            # Read header size
            header_size = struct.unpack('<Q', f.read(8))[0]
            
            # Read header
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode('utf-8').strip())
            
            # Remove metadata
            header.pop('__metadata__', None)
            
            # Read tensors
            for name, info in header.items():
                dtype = info['dtype']
                shape = info['shape']
                start_offset, end_offset = info['data_offsets']
                
                # Read tensor data
                data_size = end_offset - start_offset
                tensor_bytes = f.read(data_size)
                
                # Convert to numpy array
                if dtype == 'F32':
                    tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
                elif dtype == 'F16':
                    tensor = np.frombuffer(tensor_bytes, dtype=np.float16)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
                
                tensor = tensor.reshape(shape)
                weights[name] = tensor
        
        return weights
    
    @staticmethod
    def convert_from_hf_naming(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert HF naming back to our internal naming"""
        # Reverse mapping
        reverse_mapping = {v: k for k, v in HuggingFaceExporter.HF_KEY_MAPPING.items()}
        
        internal_weights = {}
        for hf_key, value in weights.items():
            # Check direct mapping
            if hf_key in reverse_mapping:
                internal_weights[reverse_mapping[hf_key]] = value
            # Check layer patterns
            elif hf_key.startswith('transformer.h.'):
                # Parse layer number and component
                parts = hf_key.split('.')
                if len(parts) >= 4 and parts[0] == 'transformer' and parts[1] == 'h':
                    layer_num = parts[2]
                    component_path = '.'.join(parts[3:])
                    
                    # Map back to internal naming
                    mapping = {
                        'attn.q_proj.weight': f'layer_{layer_num}.attention.q_proj.weight',
                        'attn.q_proj.bias': f'layer_{layer_num}.attention.q_proj.bias',
                        'attn.k_proj.weight': f'layer_{layer_num}.attention.k_proj.weight',
                        'attn.k_proj.bias': f'layer_{layer_num}.attention.k_proj.bias',
                        'attn.v_proj.weight': f'layer_{layer_num}.attention.v_proj.weight',
                        'attn.v_proj.bias': f'layer_{layer_num}.attention.v_proj.bias',
                        'attn.out_proj.weight': f'layer_{layer_num}.attention.out_proj.weight',
                        'attn.out_proj.bias': f'layer_{layer_num}.attention.out_proj.bias',
                        'mlp.fc_in.weight': f'layer_{layer_num}.feed_forward.linear1.weight',
                        'mlp.fc_in.bias': f'layer_{layer_num}.feed_forward.linear1.bias',
                        'mlp.fc_out.weight': f'layer_{layer_num}.feed_forward.linear2.weight',
                        'mlp.fc_out.bias': f'layer_{layer_num}.feed_forward.linear2.bias',
                        'ln_1.weight': f'layer_{layer_num}.norm1.weight',
                        'ln_1.bias': f'layer_{layer_num}.norm1.bias',
                        'ln_2.weight': f'layer_{layer_num}.norm2.weight',
                        'ln_2.bias': f'layer_{layer_num}.norm2.bias',
                    }
                    
                    if component_path in mapping:
                        internal_weights[mapping[component_path]] = value
            else:
                # Keep as is if no mapping found
                internal_weights[hf_key] = value
        
        return internal_weights


if __name__ == "__main__":
    # Example usage
    print("Hugging Face Export/Import Utilities")
    print("This module provides pure Python utilities for exporting/importing models")
    print("in Hugging Face compatible formats without using PyTorch or TensorFlow.")
    print("\nUsage:")
    print("  from huggingface_export import HuggingFaceExporter")
    print("  HuggingFaceExporter.export_model(model, tokenizer, './hf_model')")