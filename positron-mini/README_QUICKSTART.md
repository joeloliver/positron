# Tiny LLM Quickstart (Highly Commented)

This mini bundle contains:
- `bootstrap_tiny_llm.sh`: end-to-end pretraining on TinyStories with a 16k BPE tokenizer, then sampling.
- `tiny_instruct_sample.jsonl`: a toy SFT dataset (3 examples) to validate the SFT loop wiring.
- `sft_quickstart.sh`: example of instruction fine-tuning from the pretraining checkpoint.

## Why each step exists

1. **Virtual environment**: isolates dependencies so you can iterate freely.
2. **Dependencies**: PyTorch for training, SentencePiece for tokenizer, Datasets to fetch TinyStories.
3. **TinyStories**: tiny, fast corpus that lets you validate code paths and get a "feel" fast.
4. **Tokenizer**: GPT-like models expect subword tokens; BPE 16k is a good small starting point.
5. **Pretraining**: next-token prediction; you should see training loss decrease within a few hundred steps.
6. **Generation**: sanity check that the model produces text with the trained weights.
7. **SFT**: aligns behavior toward instruction following; we use a very small toy set here.

## Notes
- If the generic `pip install torch` fails, please consult https://pytorch.org/get-started/locally/ for your platform.
- For larger runs, switch to a bigger dataset (e.g., WikiText-103) and increase `--max_steps` and `batch_tokens`.
- If your code names differ, adjust paths/flags in the scripts accordingly.
