from typing import Tuple
from typing import Dict
from attr import field
from typing import List
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(): #evaluates average loss over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def clear_kv_cache(model):
    for module in model.modules():
        if isinstance(module, Head):
            module.key_cache = None
            module.value_cache = None

@dataclass
class Request():
    id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    generated_new_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"

    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=List
    )

    @property
    def tokens_so_far(self) -> List[int]:
        """Full sequence: prompt + everything generated."""
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self.num_generated >= self.max_new_tokens

class Head(nn.Module):
    """ 
    one head of self-attention 
    In single-request KV cache generation, you don't need one. 
    Every position in the cache is a real token. 
    But in continuous batching, different requests have different sequence lengths, 
    so assemble_batch_cache left-pads shorter caches with zeros to match the longest one.
    
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # self.key_cache = None
        # self.value_cache = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k = None, past_v = None, attn_mask = None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)

        if not self.training:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

                wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T_past+T, hs) @ (B, hs, T_past+T) -> (B, T_past+T, T_past+T)
                
                if attn_mask is not None:
                    new_valid = torch.ones(B, 1, T, device=wei.device, dtype=torch.bool)
                    full_mask = torch.cat([attn_mask, new_valid], dim=-1)
                    wei = wei.masked_fill(~full_mask, float('-inf'))

                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v
            else:
                # in prefill, each request processed separately
                wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v
            
            return out, k, v

        else:
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return out, None, None

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, attn_mask=None):
        """
        Args:
            x:       (B, T, C)
            past_kv: list of (past_k, past_v) per head, or None
        Returns:
            out:    (B, T, n_embd)
            new_kv: list of (new_k, new_v) per head
        """        
        if past_kv is None:
            past_kv = [(None, None) * len(self.heads)]
        
        outputs, new_kvs = [], []
        for i, h in enumerate(self.heads):
            past_keys, past_value = past_kv[i]
            out, new_keys, new_values = h(x, past_keys, past_value, attn_mask=attn_mask)
            outputs.append(out)
            new_kvs.append((new_keys, new_values))

        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_kvs

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kvs=None, attn_mask=None):
        x = x + self.sa(self.ln1(x), past_kv=past_kvs, attn_mask=attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=0, past_kvs=None, attn_mask=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        if pos is None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        else:
            pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb # (B,T,C)

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        new_kvs = []

        for i, block in enumerate(self.blocks):
            x, block_kv = block(x, past_kvs[i], attn_mask)
            new_kvs.append(block_kv)

        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # original generate function
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def generate_kv_cache(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)
    
    # Prefill: process the initial context all at once
    logits, _ = model(idx)
    
    for _ in range(max_new_tokens):
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        # Forward pass with ONLY the new token. We pass start_pos to get the right position embeddings.
        logits, _ = model(idx_next, start_pos=idx.shape[1] - 1)
        
    model.train()
    return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = block_size - context.shape[1] # total capacity - space initial prompt takes up
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# ── non-cached generate (forces full-context recompute every step) ────────────
def generate_no_cache(model, idx, max_new_tokens):
    """Runs in train mode so the KV cache branch is never entered."""
    model.train()                          # disables KV cache path
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ── cached generate (your existing path, one token fed at a time) ─────────────
def generate_with_cache(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Feed only the LAST token so the cache does the rest of the work
            logits, _ = model(idx[:, -1:])   # (B, 1, vocab_size)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ── benchmark ─────────────────────────────────────────────────────────────────
N_TOKENS   = 200
N_RUNS     = 3       # average over multiple runs for stability
context    = torch.zeros((1, 1), dtype=torch.long, device=device)

# warm-up (avoids cold-start CUDA overhead skewing results)
_ = generate_no_cache(model, context.clone(), 10)
clear_kv_cache(model)
_ = generate_with_cache(model, context.clone(), 10)

# --- No KV cache ---
times_no_cache = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    generate_no_cache(model, context.clone(), N_TOKENS)
    if device == 'cuda':
        torch.cuda.synchronize()
    times_no_cache.append(time.perf_counter() - t0)

# --- With KV cache ---
times_with_cache = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    generate_kv_cache(model, context.clone(), N_TOKENS)
    if device == 'cuda':
        torch.cuda.synchronize()
    times_with_cache.append(time.perf_counter() - t0)

avg_no_cache = sum(times_no_cache) / N_RUNS
avg_with_cache = sum(times_with_cache) / N_RUNS

print(f"Tokens generated : {N_TOKENS}")
print(f"No KV cache      : {avg_no_cache:.3f}s  ({N_TOKENS/avg_no_cache:.1f} tok/s)")
print(f"With KV cache    : {avg_with_cache:.3f}s  ({N_TOKENS/avg_with_cache:.1f} tok/s)")
print(f"Speedup          : {avg_no_cache/avg_with_cache:.2f}×")