#!/usr/bin/env python
import argparse, json, random, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import sentencepiece as spm
from gpt_min import GPT, load_config, count_params

TPL = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(p):
    return [
        json.loads(l)
        for l in Path(p).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def make_ids(ex, sp, L):
    p = TPL.format(instruction=ex.get("instruction", ""), input=ex.get("input", ""))
    a = ex.get("output", "")
    P = sp.encode(p)[: L - 2]
    A = sp.encode(a)[: L - len(P)]
    x = P + A
    y = [-100] * len(P) + A
    if len(x) < L:
        pad = [sp.pad_id() if sp.pad_id() != -1 else 0] * (L - len(x))
        x += pad
        y += [-100] * len(pad)
    return x[:L], y[:L]


def batches(ds, bs, device):
    for i in range(0, len(ds), bs):
        bx = torch.tensor(
            [ds[j][0] for j in range(i, min(i + bs, len(ds)))],
            dtype=torch.long,
            device=device,
        )
        by = torch.tensor(
            [ds[j][1] for j in range(i, min(i + bs, len(ds)))],
            dtype=torch.long,
            device=device,
        )
        yield bx, by


A = argparse.ArgumentParser()
A.add_argument("--base_ckpt", required=True)
A.add_argument("--tokenizer_path", required=True)
A.add_argument("--sft_data", required=True)
A.add_argument("--context_len", type=int, default=512)
A.add_argument("--epochs", type=int, default=2)
A.add_argument("--batch_size", type=int, default=8)
A.add_argument("--lr", type=float, default=1e-5)
A.add_argument("--warmup_steps", type=int, default=50)
A.add_argument("--out_dir", default="runs/tiny-sft")


def main():
    args = A.parse_args()
    device = dev()
    cfg = load_config(str(Path(args.base_ckpt).parent / "config.json"))
    m = GPT(cfg).to(device)
    sd = torch.load(args.base_ckpt, map_location=device)
    m.load_state_dict(sd)
    print(f"[info] params={count_params(m)/1e6:.2f}M")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    raw = read_jsonl(args.sft_data)
    random.shuffle(raw)
    split = max(1, int(0.9 * len(raw)))
    train, devset = raw[:split], raw[split:]
    train_ds = [make_ids(e, sp, args.context_len) for e in train]
    dev_ds = [make_ids(e, sp, args.context_len) for e in devset]
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    opt = optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95))
    steps = 0
    for ep in range(args.epochs):
        m.train()
        for x, y in batches(train_ds, args.batch_size, device):
            steps += 1
            logits = m(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if steps % 10 == 0:
                print(f"[epoch {ep+1}] step={steps} loss={loss.item():.3f}")
        m.eval()
        tot = n = 0
        with torch.no_grad():
            for x, y in batches(dev_ds, args.batch_size, device):
                tot += crit(m(x).view(-1, m(x).size(-1)), y.view(-1)).item()
                n += 1
        print(f"[dev] loss={tot/max(1,n):.3f}")
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ck = out / "latest.pt"
        torch.save(m.state_dict(), ck)
        print(f"[save] {ck}")


if __name__ == "__main__":
    main()
