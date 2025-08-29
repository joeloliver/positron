#!/usr/bin/env bash
# Minimal instruction fine-tuning example using the trained TinyGPT weights.
# Assumes you already ran bootstrap_tiny_llm.sh and have a checkpoint in runs/tiny-demo.
set -euo pipefail
source .venv/bin/activate

python sft_train.py \
  --config TinyGPT \
  --base_ckpt runs/tiny-demo/latest.pt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --sft_data /mnt/data/llm-bootstrap/tiny_instruct_sample.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --lr 1e-5 \
  --out_dir runs/tiny-sft

python generate.py \
  --ckpt runs/tiny-sft/latest.pt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --prompt "Explain gravity to a child." \
  --max_tokens 60 \
  --temperature 0.7
