import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32
print("using device:", device)

vocab_size = 4000     
block_size = 32       
n_layer = 6            
n_head = 6             
n_embd = 384           
dropout = 0.1          
batch_size = 4         
lr = 3e-4              
max_steps = 10000       
warmup_steps = 200

data = np.fromfile("data/bin/corpus_u16.bin", dtype=np.uint16)
data = torch.from_numpy(data.astype(np.int64))
n = int(0.9 * len(data))

train_ids, val_ids = data[:n], data[n:]
print(f"dataset loaded: {len(train_ids):,} train tokens | {len(val_ids):,} val tokens")

def get_batch(split):
    ids = train_ids if split == "train" else val_ids
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([ids[i:i+block_size] for i in ix])
    y = torch.stack([ids[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

class LayerNorm(nn.Module):
    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n))
        self.b = nn.Parameter(torch.zeros(n))
        self.eps = eps
    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, keepdim=True, unbiased=False)
        return self.g * (x - m) / torch.sqrt(v + self.eps) + self.b

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

def train():
    model = GPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def cosine(step):
        if step < warmup_steps: return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


    print("starting training loop...")
    best_val = 1e9
    os.makedirs("out-minillm", exist_ok=True)

    def estimate(split, iters=20):
        model.eval(); losses = []
        with torch.no_grad():
            for _ in range(iters):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    for step in range(1, max_steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * cosine(step)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == 1:
            val = estimate("val", iters=10)
            if val < best_val:
                best_val = val
                torch.save(
                    {"model": model.state_dict(),
                     "config": {"vocab_size": vocab_size, "block_size": block_size,
                                "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd}},
                    "out-minillm/best.pt"
                )
            print(f"step {step} | train {loss.item():.3f} | val {val:.3f}")
            

if __name__ == "__main__":
    train()

