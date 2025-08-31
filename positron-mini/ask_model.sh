#!/usr/bin/env bash
set -euo pipefail

# Positron Mini - Interactive Question Script
# Usage: ./ask_model.sh [prompt]
# If no prompt provided, script will run in interactive mode

MODEL_CKPT="runs/positron-mini-15M-v1-base/latest.pt"
TOKENIZER_PATH="tokenizer/spm.bpe.model"

# Default generation parameters (conservative for better coherence)
MAX_TOKENS=30
TEMPERATURE=0.3
TOP_P=0.8

# Colors for better UX
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if model and tokenizer exist
if [[ ! -f "$MODEL_CKPT" ]]; then
    echo "❌ Model checkpoint not found: $MODEL_CKPT"
    echo "Make sure you've trained the model first by running sft_quickstart.sh"
    exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "❌ Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Function to generate response
ask_question() {
    local prompt="$1"
    echo -e "\n${BLUE}🤖 Generating response...${NC}"
    echo -e "${YELLOW}Prompt: ${prompt}${NC}\n"
    
    python generate.py \
        --ckpt "$MODEL_CKPT" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --prompt "$prompt" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P"
}

# Main logic
if [[ $# -gt 0 ]]; then
    # Single question mode - prompt provided as argument
    prompt="$*"
    ask_question "$prompt"
else
    # Interactive mode
    echo -e "${GREEN}🚀 Positron Mini Interactive Question Mode${NC}"
    echo -e "Ask questions to your fine-tuned model!"
    echo -e "Settings: max_tokens=$MAX_TOKENS, temperature=$TEMPERATURE, top_p=$TOP_P"
    echo -e "Type 'quit' or 'exit' to stop, 'settings' to adjust parameters\n"
    
    while true; do
        echo -e -n "${BLUE}❓ Your question: ${NC}"
        read -r prompt
        
        case "$prompt" in
            "quit"|"exit"|"q")
                echo -e "${GREEN}👋 Goodbye!${NC}"
                break
                ;;
            "settings"|"config")
                echo -e "\n${YELLOW}Current settings:${NC}"
                echo "  max_tokens: $MAX_TOKENS"
                echo "  temperature: $TEMPERATURE"
                echo "  top_p: $TOP_P"
                echo -e -n "\nEnter new max_tokens (or press Enter to keep $MAX_TOKENS): "
                read -r new_max_tokens
                if [[ -n "$new_max_tokens" ]]; then
                    MAX_TOKENS="$new_max_tokens"
                fi
                echo -e -n "Enter new temperature (or press Enter to keep $TEMPERATURE): "
                read -r new_temperature
                if [[ -n "$new_temperature" ]]; then
                    TEMPERATURE="$new_temperature"
                fi
                echo -e -n "Enter new top_p (or press Enter to keep $TOP_P): "
                read -r new_top_p
                if [[ -n "$new_top_p" ]]; then
                    TOP_P="$new_top_p"
                fi
                echo -e "${GREEN}✓ Settings updated${NC}"
                continue
                ;;
            "")
                echo -e "${YELLOW}Please enter a question or 'quit' to exit${NC}"
                continue
                ;;
            *)
                ask_question "$prompt"
                echo -e "\n${GREEN}─────────────────────────────────────${NC}"
                ;;
        esac
    done
fi
