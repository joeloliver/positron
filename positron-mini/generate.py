#!/usr/bin/env python
import argparse, torch, sentencepiece as spm
from pathlib import Path
from gpt_min import GPT, load_config


def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


A = argparse.ArgumentParser()
A.add_argument("--ckpt", required=True)
A.add_argument("--tokenizer_path", required=True)
A.add_argument("--prompt", default="Hello")
A.add_argument("--max_tokens", type=int, default=80)
A.add_argument("--temperature", type=float, default=0.8)
A.add_argument("--top_p", type=float, default=0.95)


def main():
    args = A.parse_args()
    ck = Path(args.ckpt)
    cfg = load_config(str(ck.parent / "config.json"))
    device = dev()
    m = GPT(cfg).to(device)
    sd = torch.load(ck, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    ids = sp.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = m.generate(idx, args.max_tokens, args.temperature, args.top_p)
    print(sp.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
