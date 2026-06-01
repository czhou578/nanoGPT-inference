import enum
import enum
from typing import Tuple
from typing import Dict
from typing import List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from benchmarks.single_req_cont_batching import (
    RequestSpec,
    StepMetrics,
    RunMetrics,
    run_single_vs_continuous_batching_benchmark
)

# # hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# ------------

# hyperparameters for testing

batch_size = 8          # smaller training batches
block_size = 64        # keep same for now so your benchmark assumptions hold
max_iters = 120         # much faster than 5000
eval_interval = 20
learning_rate = 1e-3
device = 'cpu'          # force CPU
eval_iters = 10         # much faster validation
n_embd = 32             # was 64
n_head = 4              # 32 / 4 = 8 dim per head
n_layer = 4             # was 4
dropout = 0.0

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
            logits, loss, _ = model(X, Y)
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
            past_kv = [(None, None) for _ in range(len(self.heads))]
        
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

    def forward(self, x, past_kv=None, attn_mask=None): #here
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
    logits, _, _ = model(idx)
    
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
        logits, _, _ = model(idx_next, pos=torch.tensor([idx.shape[1] - 1], dtype=torch.long, device=device))
        
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
    logits, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = block_size - context.shape[1] # total capacity - space initial prompt takes up
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# ── 1. No KV cache (full recompute every step) ───────────────────────────────
def generate_no_cache(model, idx, max_new_tokens):
    model.train()  # disables KV cache path
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ── 2. With KV cache — passed as raw tensors ─────────────────────────────────
def generate_with_cache(model, idx, max_new_tokens):
    """KV cache stored externally — threaded through forward() each step."""
    model.eval()
    with torch.no_grad():
        # Prefill: run the entire prompt, get initial cache
        logits, _, past_kvs = model(idx)

        for step in range(max_new_tokens):
            logits = logits[:, -1, :]           # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Decode: only the new token + its position
            curr_pos = torch.tensor([[idx.shape[1] - 1]], device=device)  # (1, 1)
            logits, _, past_kvs = model(idx_next, pos=curr_pos, past_kvs=past_kvs)

    return idx

def generate_request(model: GPTLanguageModel, request: Request):
    """
    Generate for a single Request object.
    The KV cache lives on the Request, not inside the model.

    This is the building block for the continuous batching scheduler (Hint 3).
    Each request independently owns its cache, so different requests
    can have different sequence lengths and lifetimes.

    while there are active requests OR the waiting queue is non-empty:
    1. Check the waiting queue — can any new requests join the batch?
    2. Build the input tensor from ALL active requests (each contributes 1 token)
    3. Forward pass → get logits for all active requests at once
    4. Sample next token for each request
    5. Check: did any request hit its max_new_tokens? → remove it, emit its result
    6. Go to 1
    """

    model.eval()

    while torch.no_grad():
        prompt = torch.tensor(
            [request.prompt_tokens], dtype=torch.long, device=device
        )  # (1, T_prompt)

        logits, _, new_kvs = model(prompt)

        for layer_idx, block_kv in enumerate(new_kvs):
            for head_idx, (k, v) in enumerate(block_kv):
                request.kv_cache[(layer_idx, head_idx)] = (k, v)

        request.status = "active"

        while not request.is_done:
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            request.tokens_so_far.append(idx_next[0].item())

            if request.is_done: break

            past_kvs = []
            for layer_idx in range(n_layer):
                block_kv = []
                for head_idx in range(n_head):
                    block_kv.append(request.kv_cache[(layer_idx, head_idx)])
                
                past_kvs.append(block_kv)
            
            curr_pos = torch.tensor([[len(request.tokens_so_far)] - 1], device=device)

            logits, _, new_kvs = model(idx_next, pos=curr_pos, past_kvs=past_kvs)

            for layer_idx, block_kv in enumerate(new_kvs):
                for head_idx, (k, v) in enumerate(block_kv):
                    request.kv_cache[(layer_idx, head_idx)] = (k, v)
        
        request.status = "done"

def assemble_batch_cache(requests: list[Request]):
    """
    Gather per-request KV caches into batched tensors.
    LEFT-pads shorter caches so new tokens always land at the right edge.

    Big problem: You have 3 active requests. Each owns its own KV cache. You need to feed them to the model as one 
    batched tensor. But their caches have different lengths:

    Returns:
        past_kvs:    batched cache structure  [layer][head] = (B, T_max, hs)
        attn_mask:   (B, 1, T_max) bool — True = valid, False = padding
        pad_lengths: list of int — how many pad positions per request (for disassembly)
    
    Request.kv_cache = {
        # Layer 0
        (0, 0): ( Key_Tensor, Value_Tensor ), # Head 0
        (0, 1): ( Key_Tensor, Value_Tensor ), # Head 1
        (0, 2): ( Key_Tensor, Value_Tensor ), # Head 2
        (0, 3): ( Key_Tensor, Value_Tensor ), # Head 3
        
    }
    
    """

    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_len = max(lengths)

    pad_lengths = [max_len - t for t in lengths]
    
    attn_mask = torch.zeros(B, 1, max_len, dtype=torch.bool, device=device)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True
    
    past_kvs = []

    for layer_idx in range(n_layer):
        layer_kvs = []
        for head_idx in range(n_head):
            keys, values = [], []

            for i, req in enumerate(requests):
                k, v, = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=device)

                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)

                keys.append(k)
                values.append(v)
            
            layer_kvs.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        
        past_kvs.append(layer_kvs)
    
    return past_kvs, attn_mask, pad_lengths


def disassemble_batch_cache(requests, new_kvs, pad_lengths):
    """
    Scatter batched KV cache back to per-request storage.
    After Head's torch.cat, each row is (T_max + 1) — strip the left-padding.
    """

    for layer_idx, layer_kvs in enumerate(new_kvs):
        for head_idx, (k, v) in enumerate(layer_kvs):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    k[i: i + 1, pad:, :],
                    v[i: i + 1, pad:, :]
                )

def continuous_batching_generate(model, request_queue: list[Request], max_batch_size = 4):
    """
    Request objects arrive at different times:
    [(t0, reqA), (t1, reqB), (t3, reqC), (t3, reqD)]
    
    Steps:

    grab the request
    prefill it (run through model once to get kv's)
    add it to active requests

    loop until all requests done:
    
    decode the active request
    mark as done
    disassemble the kv cache (scatter back to per-request storage)

    after loop, move all active requests to completed
    return completed requests
    """

    
    model.eval()

    active_requests = []
    completed_requests = []
    step = 0
    queue_idx = 0

    with torch.no_grad():
        while active_requests or queue_idx < len(request_queue):
            while queue_idx < len(request_queue):
                time_step, req = request_queue[queue_idx]

                if time_step > step: break

                if len(active_requests) >= max_batch_size:
                    break

                prompt = torch.tensor([req.prompt_tokens], device=device)
                logits, _, new_kvs = model(prompt)

                for li, bkv in enumerate(new_kvs):
                    for hi, (k, v) in enumerate(bkv):
                        req.kv_cache[(li, hi)] = (k, v)
                
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                req.generated_tokens.append(idx_next[0].item())
            
                req.status = "active"
                req.last_token = idx_next

                if req.is_done:
                    req.status = "done"
                    completed_requests.append(req)
                else:
                    active_requests.append(req)

                queue_idx += 1

                print(f"  [step {step}] Admitted request {req.id} "
                      f"(prompt={len(req.prompt_tokens)}, "
                      f"max_new={req.max_new_tokens})")

            if not active_requests:
                step += 1
                continue

            batch_tokens = torch.cat([req.last_token for req in active_requests], dim=0)
            batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in active_requests], device=device)
            
            past_kvs, attn_mask, pad_lengths = assemble_batch_cache(active_requests)

            logits, _, new_kvs = model(batch_tokens, pos=batch_positions, past_kvs=past_kvs, attn_mask=attn_mask)                    

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            disassemble_batch_cache(active_requests, new_kvs, pad_lengths)

            for i, req in enumerate(active_requests):
                req.generated_tokens.append(idx_next[i, 0].item())
                req.last_token = idx_next[i:i+1]

            still_active = []

            for req in active_requests:
                if not req.is_done:
                    still_active.append(req)
                else:
                    req.status = "done"
                    completed_requests.append(req)
            
            active_requests = still_active
            step += 1
    
    return completed_requests
            
# run_single_vs_continuous_batching_benchmark(
#     model,
#     vocab_size=vocab_size,
#     num_requests=16,
#     prompt_len=8,
#     max_new_tokens=32,
#     max_batch_size=8,
#     arrival_gap=0,
#     device=device,
# )

run_single_vs_continuous_batching_benchmark(
    m, vocab_size=vocab_size,
    num_requests=32,
    prompt_len=8,
    max_new_tokens=24,
    max_batch_size=8,
    arrival_gap=0,
    device=device,
)

# num_requests=16
# prompt_len=8
# max_new_tokens=32
# max_batch_size=8

    



                





