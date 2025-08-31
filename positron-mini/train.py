#!/usr/bin/env python
import argparse, math, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim, sentencepiece as spm
from gpt_min import GPT, GPTConfig, save_config, count_params


def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_of(p):
    p = p.lower()
    return (
        torch.bfloat16
        if p in ("bf16", "bfloat16")
        else (torch.float16 if p in ("fp16", "float16") else torch.float32)
    )


def cosine_lr(it, max_it, warmup, base):
    if it < warmup:
        return base * (it + 1) / max(1, warmup)
    prog = (it - warmup) / max(1, max_it - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog)) * base


def load_text_tokens(path, sp):
    return torch.tensor(
        sp.encode(Path(path).read_text(encoding="utf-8")), dtype=torch.long
    )


def iter_batches(ids, L, bs, device):
    starts = list(range(0, len(ids) - (L + 1), L))
    random.shuffle(starts)
    for i in range(0, len(starts), bs):
        ss = starts[i : i + bs]
        x = torch.stack([ids[s : s + L] for s in ss]).to(device)
        y = torch.stack([ids[s + 1 : s + L + 1] for s in ss]).to(device)
        yield x, y


A = argparse.ArgumentParser()
A.add_argument("--config", default="TinyGPT")
A.add_argument("--dataset_path", required=True)
A.add_argument("--tokenizer_path", required=True)
A.add_argument("--context_len", type=int, default=512)
A.add_argument("--batch_tokens", type=int, default=65536)
A.add_argument("--max_steps", type=int, default=800)
A.add_argument("--warmup_steps", type=int, default=100)
A.add_argument("--lr", type=float, default=3e-4)
A.add_argument("--weight_decay", type=float, default=0.1)
A.add_argument("--precision", default="bf16")
A.add_argument("--grad_clip", type=float, default=1.0)
A.add_argument("--save_every", type=int, default=200)
A.add_argument("--out_dir", default="runs/tiny-demo")
A.add_argument("--resume_from_step", type=int, default=0)


def main():
    args = A.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    V = sp.get_piece_size()
    cfg = (
        GPTConfig.tiny(V, args.context_len)
        if args.config.lower() == "tinygpt"
        else GPTConfig.mini(V, args.context_len)
    )
    save_config(cfg, out / "config.json")
    device = dev()
    dtype = dtype_of(args.precision)
    print(f"[info] device={device}, dtype={dtype}")
    model = GPT(cfg).to(device)
    print(f"[info] params={count_params(model)/1e6:.2f}M")
    opt = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    ids = load_text_tokens(args.dataset_path, sp)
    bs = max(1, args.batch_tokens // args.context_len)
    print(f"[info] seq_len={args.context_len}, batch_size={bs}")
    use_amp = dtype in (torch.float16, torch.bfloat16)
    # Use appropriate scaler for device type
    use_scaler = dtype == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    
    # Resume from checkpoint if specified
    start_step = args.resume_from_step
    if start_step > 0 and (out / "latest.pt").exists():
        print(f"[info] Resuming from step {start_step}, loading {out / 'latest.pt'}")
        model.load_state_dict(torch.load(out / "latest.pt", map_location=device))
    
    step, t0 = start_step, time.time()
    while step < args.max_steps:
        for x, y in iter_batches(ids, args.context_len, bs, device):
            step += 1
            lr = cosine_lr(step - 1, args.max_steps, args.warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            device_type = str(device).split(":")[0]
            use_autocast = device_type in ["cuda", "mps"] and dtype in [torch.float16, torch.bfloat16]
            
            if use_amp and use_autocast:
                # Use autocast + scaler for mixed precision training
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                # Standard training without mixed precision
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            # Show every step for real-time monitoring
            if True:
                tokps = (bs * args.context_len) / max(1e-6, (time.time() - t0))
                print(
                    f"[step {step}/{args.max_steps}] loss={loss.item():.3f} lr={lr:.2e} ~{tokps:.0f} tok/s"
                )
                t0 = time.time()
            if step % args.save_every == 0 or step == args.max_steps:
                # Save both latest.pt and step-numbered checkpoint
                ck_latest = out / "latest.pt"
                ck_step = out / f"step_{step}.pt"
                torch.save(model.state_dict(), ck_latest)
                torch.save(model.state_dict(), ck_step)
                print(f"[save] {ck_latest}")
                print(f"[save] {ck_step}")
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
