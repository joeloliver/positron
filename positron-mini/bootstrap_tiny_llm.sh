#!/usr/bin/env bash
# ============================================================================
# bootstrap_tiny_llm.sh
# ----------------------------------------------------------------------------
# One-shot script to:
#   1) Create a Python virtual environment
#   2) Install minimal deps (PyTorch, SentencePiece, Datasets, tqdm)
#   3) Download a tiny corpus (TinyStories) for quick experiments
#   4) Train a 16k SentencePiece tokenizer on that corpus
#   5) Kick off a short pretraining run of a TinyGPT (~15M params)
#   6) Generate a sample from the resulting checkpoint
#
# This is designed to give you the "feel" of the full pipeline FAST, without
# caring about perfect quality. Lots of comments are included to explain WHY
# each step exists.
#
# Assumptions:
# - You have cloned the "llm-from-scratch" repo (the one with train.py, etc.)
# - You run this script from the root of that repo, e.g.:
#       bash /path/to/bootstrap_tiny_llm.sh
#
# If you named your repo differently, adjust the paths below.
# ============================================================================
set -euo pipefail

# ----[ 0. Detect OS (for PyTorch install hint) ]------------------------------
OS_NAME=$(uname -s || echo "Unknown")
echo "[info] Detected OS: ${OS_NAME}"
echo "[info] If PyTorch install fails, see: https://pytorch.org/get-started/locally/"

# ----[ 1. Python venv ]------------------------------------------------------
# WHY: Keep dependencies isolated and reproducible. You can nuke the venv
# anytime without touching your system Python.
if [ ! -d ".venv" ]; then
  echo "[step] Creating virtual environment .venv"
  python3 -m venv .venv
fi
# Activate for this shell
# shellcheck disable=SC1091
source .venv/bin/activate
python -V

# ----[ 2. Dependencies ]-----------------------------------------------------
# WHY: Minimal set to run the pipeline.
# - torch: deep learning engine
# - sentencepiece: tokenizer training + inference
# - datasets: easy dataset download (TinyStories) from Hugging Face
# - tqdm: progress bars
echo "[step] Installing Python dependencies"
pip install --upgrade pip wheel setuptools

# We try a generic torch install first. If it fails, point the user to the URL.
# For CUDA users, you may prefer a CUDA-specific index-url. Examples below.
pip install torch --index-url https://download.pytorch.org/whl/cpu || true

# If you have NVIDIA GPU and CUDA 12.1, uncomment this line instead:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# If you're on Apple Silicon (MPS), the generic CPU wheel is fine; enable MPS at runtime.

pip install sentencepiece datasets tqdm

# ----[ 3. Data prep: TinyStories ]-------------------------------------------
# WHY: Tiny corpus to iterate quickly (fast downloads, fast tokenizer training).
# We'll save a single plain-text file at data/tinystories.txt.
echo "[step] Preparing data directory"
mkdir -p data

python - << 'PY'
import pathlib, sys
from datasets import load_dataset

out_path = pathlib.Path("data/tinystories.txt")
if out_path.exists():
    print(f"[info] {out_path} already exists; skipping download.")
    sys.exit(0)

print("[step] Downloading TinyStories (validation split for speed)")
# WHY 'validation'?: it's small and enough to validate the pipeline end-to-end.
# For a slightly larger run, use split='train' or concatenate both.
ds = load_dataset("roneneldan/TinyStories", split="validation")
# Concatenate stories with blank lines. Real training would need better cleaning.
text = "\n\n".join(ex["text"] for ex in ds if "text" in ex)
out_path.write_text(text, encoding="utf-8")
print(f"[done] Wrote {out_path} with {len(text):,} characters.")
PY

# ----[ 4. Train tokenizer (SentencePiece BPE, 16k vocab) ]-------------------
# WHY: Modern LLMs rely on subword tokenizers; we learn the merges on *our*
# small corpus to keep things consistent and compact.
echo "[step] Training SentencePiece tokenizer (BPE, vocab=16000)"
python - << 'PY'
import sentencepiece as spm
from pathlib import Path

Path("tokenizer").mkdir(exist_ok=True)
# --model_type=bpe : common choice for GPT-like models
# --character_coverage=1.0 : include all chars we see (enough for tiny demo)
# --model_prefix : output files tokenizer/spm.bpe.{model,vocab}
spm.SentencePieceTrainer.Train(
    input="data/tinystories.txt",
    model_prefix="tokenizer/spm.bpe",
    vocab_size=16000,
    model_type="bpe",
    character_coverage=1.0,
    input_sentence_size=2000000, # shuffle subset cap (sane default)
    shuffle_input_sentence=True
)
print("[done] Tokenizer artifacts at tokenizer/spm.bpe.model and .vocab")
PY

# ----[ 5. Tiny pretraining run ]---------------------------------------------
# WHY: Next-token prediction pretraining teaches the model to continue text.
# We keep it tiny and short so you can *see* progress quickly.
#
# We assume your repo exposes:
#   - model/config.py      (TinyGPT-15M config)
#   - train.py             (pretraining loop)
#
# Adjust flags below to match your code if needed.
echo "[step] Starting a SHORT TinyGPT pretraining run"
python train.py \
  --config TinyGPT \
  --dataset_path data/tinystories.txt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --context_len 512 \
  --batch_tokens 65536 \
  --max_steps 800 \
  --warmup_steps 100 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --precision bf16 \
  --grad_clip 1.0 \
  --save_every 200 \
  --out_dir runs/tiny-demo

# Notes:
# - batch_tokens ~ how many tokens per optimization step (grad accumulation OK)
# - max_steps kept tiny (800) so this finishes quickly on CPU/GPU
# - precision bf16: if not supported, your code should fall back to fp32/fp16
# - save_every: periodic checkpoints (useful in case you interrupt)

# ----[ 6. Generate a sample ]------------------------------------------------
# WHY: Sanity-check that the model can produce text at all.
# We load the latest checkpoint from runs/tiny-demo and sample a continuation.
echo "[step] Generating a quick sample"
python generate.py \
  --ckpt runs/tiny-demo/latest.pt \
  --tokenizer_path tokenizer/spm.bpe.model \
  --prompt "Once upon a time" \
  --max_tokens 80 \
  --temperature 0.8 \
  --top_p 0.95

echo ""
echo "============================================================================"
echo "[success] End-to-end pipeline finished."
echo "Try changing dataset_path to something bigger (e.g., WikiText-103) or"
echo "increasing --max_steps to see longer training."
echo "============================================================================"
