#!/usr/bin/env python
from dataclasses import dataclass, asdict
import json, math, torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class GPTConfig:
    d_model: int = 256
    n_layer: int = 8
    n_head: int = 8
    d_ff: int = 1024
    vocab_size: int = 16000
    context_len: int = 512
    dropout: float = 0.0
    tie_weights: bool = True

    @staticmethod
    def tiny(vocab_size: int, context_len: int = 512):
        return GPTConfig(vocab_size=vocab_size, context_len=context_len)

    @staticmethod
    def mini(vocab_size: int, context_len: int = 1024):
        return GPTConfig(
            d_model=768,
            n_layer=12,
            n_head=12,
            d_ff=3072,
            vocab_size=vocab_size,
            context_len=context_len,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.n_head
        D = cfg.d_model // H

    def __init__(self, cfg):
        super().__init__()
        self.H = cfg.n_head
        self.D = cfg.d_model // cfg.n_head
        self.L = cfg.context_len
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_len, cfg.context_len)).view(
                1, 1, cfg.context_len, cfg.context_len
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        H = self.H
        D = self.D
        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.att = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None, :, :]
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
            x = self.ln_f(x)
            return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_p=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len :]
            logits = self(idx_cond)[:, -1, :] / max(1e-5, temperature)
            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sp = sp.masked_fill(mask, 0.0)
                sp = sp / sp.sum(dim=-1, keepdim=True)
                nid = torch.multinomial(sp, 1)
                next = si.gather(-1, nid)
            else:
                next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next], dim=1)
        return idx


def save_config(cfg, path):
    import json
    from dataclasses import asdict

    open(path, "w").write(json.dumps(asdict(cfg), indent=2))


def load_config(path):
    d = json.loads(open(path).read())
    return GPTConfig(**d)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
