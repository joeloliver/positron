#!/usr/bin/env bash
set -euo pipefail

#Might need to run 
cp runs/positron-mini-15M-v1-base/config.json runs/positron-mini-15M-v1-sft/config.json

MODEL_ID="positron-mini-15M-v1"
BASE_DIR="runs/${MODEL_ID}-base"
SFT_DIR="runs/${MODEL_ID}-sft"

# shellcheck disable=SC1091
source .venv/bin/activate

python sft_train.py \
  --base_ckpt "${BASE_DIR}/latest.pt" \
  --tokenizer_path tokenizer/spm.bpe.model \
  --sft_data tiny_instruct_sample.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --lr 1e-5 \
  --out_dir "${SFT_DIR}"

python generate.py \
  --ckpt "${SFT_DIR}/latest.pt" \
  --tokenizer_path tokenizer/spm.bpe.model \
  --prompt "Explain gravity to a child." \
  --max_tokens 60 \
  --temperature 0.7
