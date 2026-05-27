"""
Chunked Prefill

"""

import enum
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)  # unpack 3 return values now
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@dataclass
class Request:
    """Each in-flight generation carries its own state and KV cache."""
    id: int
    prompt_tokens: List[int]          # the original encoded prompt
    max_new_tokens: int               # how many tokens this request wants
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"           # "waiting" -> "prefilling" -> "active" -> "done"
    prefill_cursor: int = 0           # how many prompt tokens we've already processed

    # Hint 2: Per-request KV cache, keyed by (layer_idx, head_idx)
    # Each value is a (key_tensor, value_tensor) tuple of shape (1, T_i, head_size)
    # T_i grows by 1 each decode step — different requests have different T_i
    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
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
    
    @property
    def is_fully_prefilled(self) -> bool:
        return self.prefill_cursor == len(self.prompt_tokens)

    def clear_cache(self):
        self.kv_cache.clear()


class Head(nn.Module):
    """One head of self-attention — now STATELESS (no internal cache)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None, attn_mask=None):
        """
        Args:
            x:      (B, T, C)       input embeddings
            past_k: (B, T_past, hs) cached keys, or None
            past_v: (B, T_past, hs) cached values, or None
        Returns:
            out:   (B, T, hs)           attention output
            new_k: (B, T_past+T, hs)    updated key cache   (None during training)
            new_v: (B, T_past+T, hs)    updated value cache  (None during training)
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        if not self.training:
            if past_k is not None:
                # ── Decode step: append new K/V onto cached past ──
                k = torch.cat([past_k, k], dim=1)  # (B, T_past + T, hs)
                v = torch.cat([past_v, v], dim=1)

                # Q attends over full cache — no causal mask needed (T=1)
                wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

                if attn_mask is not None:
                    new_valid = torch.ones(B, 1, T, device=wei.device, dtype=torch.bool)
                    full_mask = torch.cat([attn_mask, new_valid], dim=-1)
                    wei = wei.masked_fill(~full_mask, float('-inf'))

                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v
            else:
                # ── Prefill step: full prompt, needs causal mask ──
                wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v

            return out, k, v   # return updated cache
        else:
            # ── Training path — unchanged, no cache ──
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, None, None


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

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
            past_kv = [(None, None)] * len(self.heads)

        outputs, new_kvs = [], []
        for i, h in enumerate(self.heads):
            pk, pv = past_kv[i]
            out, nk, nv = h(x, pk, pv, attn_mask=attn_mask)
            outputs.append(out)
            new_kvs.append((nk, nv))

        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_kvs


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

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
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kv=None, attn_mask=None):
        """
        Returns:
            x:      (B, T, n_embd)
            new_kv: list of (new_k, new_v) per head in this block
        """
        sa_out, new_kv = self.sa(self.ln1(x), past_kv, attn_mask=attn_mask)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # ModuleList instead of Sequential so we can pass per-block cache
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=None, past_kvs=None, attn_mask=None):
        """
        Args:
            idx:      (B, T) token indices
            targets:  (B, T) target indices, or None
            pos:      (B, T) explicit position indices, or None (uses arange)
            past_kvs: list-of-lists cache structure, or None
                      past_kvs[layer][head] = (key_tensor, value_tensor)
        Returns:
            logits:   (B, T, vocab_size)
            loss:     scalar or None
            new_kvs:  updated cache with same structure as past_kvs
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        if pos is None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        else:
            pos_emb = self.position_embedding_table(pos)  # (B, T, C)

        x = tok_emb + pos_emb  # (B, T, C)

        # Thread cache through each block
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, block_kv = block(x, past_kvs[i], attn_mask=attn_mask)
            new_kvs.append(block_kv)

        x = self.ln_f(x)          # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kvs

    def generate(self, idx, max_new_tokens):
        """Original generate (no cache, full recompute) for reference."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss, _ = model(xb, yb)  # _ discards the cache during training
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Quick sanity check with the original no-cache generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

def assemble_batch_cache(requests):
    """
    Gather per-request KV caches into batched tensors.
    LEFT-pads shorter caches so new tokens always land at the right edge.

    Big problem: You have 3 active requests. Each owns its own KV cache. You need to feed them to the model as one
    batched tensor. But their caches have different lengths:

    Returns:
        past_kvs:    batched cache structure  [layer][head] = (B, T_max, hs)
        attn_mask:   (B, 1, T_max) bool — True = valid, False = padding
        pad_lengths: list of int — how many pad positions per request (for disassembly)
    """

    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_t = max(lengths)

    pad_lengths = [max_t - t for t in lengths] # pad lengths for every position in t

    attn_mask = torch.zeros(B, 1, max_t, device=device, dtype=torch.bool)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, 0, pad:] = True

    past_kvs = []

    for layer_idx in range(n_layer):
        block_kv = []

        for head_idx in range(n_head):
            keys, values = [], []

            for i, req in enumerate(requests):
                k, v = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=device)
                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)

                keys.append(k)
                values.append(v)

            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths

def disassemble_batch_cache(requests, new_kvs, pad_lengths):
    """
    Scatter batched KV cache back to per-request storage.
    After Head's torch.cat, each row is (T_max + 1) — strip the left-padding.
    """
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    batched_k[i : i + 1, pad:, :],      # (1, T_i + 1, hs)
                    batched_v[i : i + 1, pad:, :],
                )

def chunked_prefill_generate(model, request_queue: list[Request], token_budget: int = 16):
    """
    Step 1: Initialize
    Create empty lists: active_requests, done_requests, all_generated_tokens.

    Step 2: Main Loop
    Loop while request_queue is not empty OR active_requests is not empty.

    Step 3: Budget Calculation
    Remaining budget = token_budget - len(active_requests). (Decode requests take priority).

    Step 4: Admit New Requests
    While there are waiting requests and remaining_budget > 0:
        Move a request from request_queue to active_requests.
        Set its status to "prefilling".
        Decrement remaining_budget.
        Break if budget runs out.

    Step 5: Prefill Phase (Unfused)
    Identify the single request currently in "prefilling" state.
    If no request is prefilling, skip this block.
    Calculate chunk_size = min(remaining_budget, prompt_tokens_left).
    Extract the chunk of input tokens and create the position tensor.
    Call model(chunk_tokens, pos=chunk_pos, past_kvs=current_kv_cache).
    Save the returned new_kvs back to the request's kv_cache.
    Update prefill_cursor.
    If prefill is now complete:
        Take the last logit from the chunk.
        Sample the first generated token.
        Append it to that request's generated_tokens.
        Set status to "active".

    Step 6: Decode Phase (Unfused)
    Filter active_requests to only those in "active" state.
    If there are any active requests:
        Assemble their KV caches into batched tensors using assemble_batch_cache.
        Run model(active_tokens, past_kvs=batched_kvs).
        Disassemble the new KV caches back to per-request using disassemble_batch_cache.
        For each request:
            Take the last logit.
            Sample the next token.
            Append it to generated_tokens.

    Step 7: Housekeeping
    Identify any requests that are now "done" (e.g. reached max_new_tokens).
    Remove them from active_requests and add them to done_requests.

    Step 8: Advance Time
    Increment the step counter.

    Step 9: Termination
    Once the loop finishes, return all_generated_tokens.
    """

    model.eval()
    active_requests = []
    completed_requests = []
    prefilling_requests = []
    step = 0
    queue_idx = 0

    with torch.no_grad():
        while queue_idx < len(request_queue) or active_requests or prefilling_requests:

            prefill_chunk_tokens = None # what we feed into model
            remaining_budget = token_budget - len(active_requests)

            if not prefilling_requests and queue_idx < len(request_queue):
                time_step, new_req = request_queue[queue_idx]
                if time_step <= step:
                    new_req.status = "prefilling"
                    prefilling_requests.append(new_req)
                    queue_idx += 1

            if remaining_budget > 0 and prefilling_requests:
                p_req = prefilling_requests[0]
                tokens_left = len(p_req.prompt_tokens) - p_req.prefill_cursor
                chunk_size = min(remaining_budget, tokens_left) # chunk_end equivalent

                chunk_start = p_req.prefill_cursor
                chunk_tokens = p_req.prompt_tokens[chunk_start : chunk_start + chunk_size]

                prefill_chunk_tokens = torch.tensor([chunk_tokens], device=device)

                p_req.prefill_cursor += chunk_size

            if prefill_chunk_tokens is None and not active_requests:
                step += 1
                continue
                
            if prefill_chunk_tokens is not None:
                pos = torch.arange(chunk_start, chunk_start + chunk_size, device=device).unsqueeze(0)

                if p_req.kv_cache:
                    past_kvs = []

                    for layer_idx in range(n_layer):
                        block_kv = []
                        for head_idx in range(n_head):
                            k, v = p_req.kv_cache[(layer_idx, head_idx)]
                            block_kv.append((k, v))

                        past_kvs.append(block_kv)

                    logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos, past_kvs=past_kvs)
                
                else:
                    logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos)
                
                for li, bkv in enumerate(new_kvs):
                    for hi, (k, v) in enumerate(bkv):
                        p_req.kv_cache[(li, hi)] = (k, v)
                
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                if p_req.is_fully_prefilled:
                    p_req.generated_tokens.append(idx_next.item())
                    p_req.status = "active"
                    p_req.last_token = idx_next
                    active_requests.append(p_req)
                    prefilling_requests.pop(0)
            
            # decode phase
            if active_requests:
                batch_tokens = torch.cat([req.last_token for req in active_requests])
                batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in active_requests], device=device)
                
                past_kvs, attn_mask, pad_lengths = assemble_batch_cache(active_requests)
                logits, _, new_kvs = model(batch_tokens, pos=batch_positions, past_kvs=past_kvs, attn_mask=attn_mask)

                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                disassemble_batch_cache(active_requests, new_kvs, pad_lengths)

                for i, req in enumerate(active_requests):
                    req.generated_tokens.append(idx_next[i].item())
                    req.last_token = idx_next[i : i + 1]
                
            still_active = []

            for req in active_requests:
                if req.is_done:
                    completed_requests.append(req)
                else:
                    still_active.append(req)
            
            active_requests = still_active

        step += 1
        
    return completed_requests
                            

