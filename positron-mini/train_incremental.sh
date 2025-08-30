#!/usr/bin/env bash
set -euo pipefail

# Incremental Training Script - Train in 1K step chunks
# Usage: ./train_incremental.sh [step_chunk_size]

MODEL_ID="positron-mini-15M-v1"
BASE_DIR="runs/${MODEL_ID}-base"
STEP_CHUNK=${1:-1000}  # Default 1K steps per chunk
TOTAL_STEPS=5000

# Colors for better UX
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activate virtual environment
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Optimize for M3 Max Apple Silicon (same as bootstrap script)
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
    echo -e "${BLUE}[info] M3 Max detected - using Apple Silicon optimizations${NC}"
fi
echo -e "${BLUE}[info] Using precision=${PRECISION}, batch_tokens=${BATCH_TOKENS}${NC}"

# Function to get current step from latest checkpoint
get_current_step() {
    if [[ -f "${BASE_DIR}/latest.pt" ]]; then
        # Try to extract step from checkpoint filename pattern
        local step_files=(${BASE_DIR}/step_*.pt)
        if [[ ${#step_files[@]} -gt 0 ]]; then
            local latest_step_file=$(ls -1 ${BASE_DIR}/step_*.pt 2>/dev/null | tail -1)
            if [[ -n "$latest_step_file" ]]; then
                basename "$latest_step_file" | sed 's/step_\([0-9]*\)\.pt/\1/'
                return
            fi
        fi
        echo "0"  # Fallback if we can't determine step
    else
        echo "0"
    fi
}

# Function to train for a chunk of steps
train_chunk() {
    local start_step=$1
    local end_step=$2
    local steps_to_train=$((end_step - start_step))
    
    echo -e "\n${YELLOW}🚀 Training steps ${start_step} → ${end_step} (${steps_to_train} steps)${NC}"
    
    python train.py \
        --config TinyGPT \
        --dataset_path data/tinystories.txt \
        --tokenizer_path tokenizer/spm.bpe.model \
        --context_len 512 \
        --batch_tokens "${BATCH_TOKENS}" \
        --max_steps "${end_step}" \
        --warmup_steps 500 \
        --lr 3e-4 \
        --weight_decay 0.1 \
        --precision "${PRECISION}" \
        --grad_clip 1.0 \
        --save_every 200 \
        --out_dir "${BASE_DIR}" \
        --resume_from_step "${start_step}"
}

# Function to test model at current checkpoint
test_model() {
    local current_step=$1
    echo -e "\n${BLUE}🧪 Testing model at step ${current_step}...${NC}"
    
    python generate.py \
        --ckpt "${BASE_DIR}/latest.pt" \
        --tokenizer_path tokenizer/spm.bpe.model \
        --prompt "Once upon a time, there was a little girl who" \
        --max_tokens 40 \
        --temperature 0.5
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Main incremental training loop
echo -e "${GREEN}🎯 Incremental Training: ${STEP_CHUNK} steps per chunk, ${TOTAL_STEPS} total steps${NC}"

mkdir -p "${BASE_DIR}"

current_step=$(get_current_step)
echo -e "${BLUE}[info] Starting from step: ${current_step}${NC}"

while [[ $current_step -lt $TOTAL_STEPS ]]; do
    next_step=$((current_step + STEP_CHUNK))
    if [[ $next_step -gt $TOTAL_STEPS ]]; then
        next_step=$TOTAL_STEPS
    fi
    
    # Train the chunk
    train_chunk $current_step $next_step
    
    # Test the model
    test_model $next_step
    
    # Ask user if they want to continue
    if [[ $next_step -lt $TOTAL_STEPS ]]; then
        echo -e -n "${YELLOW}Continue training to step $((next_step + STEP_CHUNK))? [Y/n]: ${NC}"
        read -r response
        if [[ "$response" =~ ^[Nn] ]]; then
            echo -e "${RED}Training stopped by user at step ${next_step}${NC}"
            break
        fi
    fi
    
    current_step=$next_step
done

echo -e "\n${GREEN}✅ Incremental training complete! Final model at step ${current_step}${NC}"
echo -e "${BLUE}You can now run: ./ask_model.sh${NC}"
