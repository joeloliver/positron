#!/usr/bin/env bash
# ask_model.sh
PROMPT="### Instruction:
$1

### Input:

### Response:
"
. .venv/bin/activate
python generate.py \
  --ckpt runs/positron-mini-15M-v1-sft/latest.pt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --prompt "$PROMPT" \
  --max_tokens 120 --temperature 0.5 --top_p 0.9