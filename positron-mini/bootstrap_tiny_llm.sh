#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="positron-mini-15M-v1"
BASE_DIR="runs/${MODEL_ID}-base"

OS_NAME="$(uname -s || echo Unknown)"
ARCH="$(uname -m || echo Unknown)"
echo "[info] OS=${OS_NAME} ARCH=${ARCH}"

if [ ! -d ".venv" ]; then
  echo "[step] Creating virtual environment .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -V
pip install --upgrade pip wheel setuptools

if [[ "${OS_NAME}" == "Darwin" ]]; then
  echo "[step] Installing torch (macOS)"
  pip install --upgrade torch
else
  echo "[step] Installing torch (Linux; CPU wheel by default)"
  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu || true
  # CUDA example:
  # pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
fi
pip install --upgrade sentencepiece datasets tqdm

python - << 'PY'
import torch
dev = ("cuda" if torch.cuda.is_available() else
       ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
print(f"[info] torch version: {torch.__version__}, device: {dev}")
PY

mkdir -p data
python - << 'PY'
from datasets import load_dataset
from pathlib import Path
out = Path("data/tinystories.txt")
if out.exists():
    print("[info] data/tinystories.txt exists; skipping")
else:
    print("[step] Downloading TinyStories (validation split)")
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    text = "\n\n".join(ex["text"] for ex in ds if "text" in ex)
    out.write_text(text, encoding="utf-8")
    print(f"[done] Wrote {out} ({len(text):,} chars)")
PY

python - << 'PY'
import sentencepiece as spm
from pathlib import Path
Path("tokenizer").mkdir(exist_ok=True)
if Path("tokenizer/spm.bpe.model").exists():
    print("[info] tokenizer/spm.bpe.model exists; skipping")
else:
    print("[step] Training SentencePiece BPE (vocab=16000)")
    spm.SentencePieceTrainer.Train(
        input="data/tinystories.txt",
        model_prefix="tokenizer/spm.bpe",
        vocab_size=16000, model_type="bpe", character_coverage=1.0,
        input_sentence_size=2000000, shuffle_input_sentence=True
    )
    print("[done] Tokenizer at tokenizer/spm.bpe.model")
PY

# Optimize for M3 Max Apple Silicon
PRECISION_AND_BATCH=$(python - << 'PY'
import torch
if torch.cuda.is_available():
    print("bf16 131072")  # CUDA: use bf16 + larger batch
elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
    print("bf16 98304")   # M3 Max: bf16 works well + optimized batch size for unified memory
else:
    print("fp32 65536")   # CPU fallback
PY
)
PRECISION=$(echo $PRECISION_AND_BATCH | cut -d' ' -f1)
BATCH_TOKENS=$(echo $PRECISION_AND_BATCH | cut -d' ' -f2)

if [[ "$PRECISION" == "bf16" ]] && python -c "import torch; exit(0 if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "[info] M3 Max detected - enabling Apple Silicon optimizations"
    echo "[info] Using bfloat16 precision with MPS backend"
    echo "[info] Batch tokens optimized for unified memory: ${BATCH_TOKENS}"
fi
echo "[info] Using precision=${PRECISION}, batch_tokens=${BATCH_TOKENS}"

python train.py \
  --config TinyGPT \
  --dataset_path data/tinystories.txt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --context_len 512 \
  --batch_tokens "${BATCH_TOKENS}" \
  --max_steps 5000 \
  --warmup_steps 500 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --precision "${PRECISION}" \
  --grad_clip 1.0 \
  --save_every 200 \
  --out_dir "${BASE_DIR}"

python generate.py \
  --ckpt "${BASE_DIR}/latest.pt" \
  --tokenizer_path tokenizer/spm.bpe.model \
  --prompt "Once upon a time" \
  --max_tokens 80 \
  --temperature 0.8 \
  --top_p 0.95

echo ""
echo "============================================================================"
echo "[success] positron-mini-15M-v1-base is ready at ${BASE_DIR}"
echo "Next: bash sft_quickstart.sh  (to create positron-mini-15M-v1-sft)"
echo "============================================================================"
