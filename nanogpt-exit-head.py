"""
NanoGPT + Early Exit Heads.

Adds confidence-based early exit to the KV-cached transformer. Exit heads
(lightweight linear classifiers) are attached after each transformer block.
During decode, if an exit head's top-token probability exceeds a threshold,
the model returns early - skipping deeper layers for that token.

Builds on: nanogpt-kv-cache.py
Key additions:
    - ExitHead module (LayerNorm + Linear) after each block
    - Joint training loss (final + alpha * sum of exit losses)
    - Confidence-gated early exit during inference
    - KV cache backfill for skipped layers
    - Per-layer exit statistics tracking

See: notes/plans/early-exit-plan.md

Run:
    python nanogpt-exit-head.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


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
            logits, loss, _, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def clear_kv_cache(model):
    for module in model.modules():
        if isinstance(module, Head):
            module.key_cache = None
            module.value_cache = None

def compute_joint_loss(model, idx, targets):
    """
    Run the full model, collect exit head logits, and compute
    the joint loss = final_loss + sum(alpha * exit_losses).
    """

    logits, _, _, all_exit_logits = model(idx, targets=targets)
    B, T, C = logits.shape

    loss_ce = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    
    alpha = 0.3

    exit_loss = 0.0

    for exit_logits in all_exit_logits:
        exit_loss += F.cross_entropy(
            exit_logits.view(B*T, C), targets.view(B*T)
        )
    
    loss = loss_ce + alpha * exit_loss
    return loss

def generate_early_exit(model, idx, max_new_tokens, exit_threshold=0.9):
    model.eval()
    clear_kv_cache(model)

    exit_counts = [0] * n_layer

    logits, _, _, _ = model(idx)

    for step in range(max_new_tokens):
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        logits, _, exit_layer, _ = model(
            idx_next,
            start_pos=idx.shape[1] - 1,
            exit_threshold=exit_threshold,
        )
        
        exit_counts[exit_layer] += 1

    model.train()

    # Print exit statistics
    total = sum(exit_counts)
    print("\n--- Early Exit Statistics ---")
    for layer, count in enumerate(exit_counts):
        label = f"Layer {layer}" if layer < n_layer - 1 else f"Layer {layer} (full)"
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {label}: {count:4d} tokens ({pct:5.1f}%)")

    return idx

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.key_cache = None
        self.value_cache = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)

        if not self.training:
            if self.key_cache is not None:
                self.key_cache = torch.cat([self.key_cache, k], dim=-2)
                self.value_cache = torch.cat([self.value_cache, v], dim=-2)
            else:
                self.key_cache = k
                self.value_cache = v
            
            wei = q @ self.key_cache.transpose(-2, -1) * (self.key_cache.shape[-1] ** -0.5)
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ self.value_cache

            return out 

        else:
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ExitHead(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.linear = nn.Linear(n_embd, vocab_size)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.linear(self.ln(x))

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.exit_heads = nn.ModuleList([ExitHead(n_embd, vocab_size) for _ in range(n_layer - 1)])
        
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

    def forward(self, idx, targets=None, start_pos=0, exit_threshold=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_ids = torch.arange(start_pos, start_pos + T, device=device).clamp(max=block_size - 1)
        pos_emb = self.position_embedding_table(pos_ids) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        all_exit_logits = []

        for layer_idx, block in enumerate(self.blocks):
            x = block(x)

            if layer_idx < n_layer - 1:
                exit_logits = self.exit_heads[layer_idx](x)

                if self.training:
                    all_exit_logits.append(exit_logits)
                
                elif exit_threshold is not None:
                    probs = F.softmax(exit_logits[:, -1, :], dim=-1)    # (B, vocab_size)
                    confidence = probs.max(dim=-1).values.item()
                    if confidence > exit_threshold:
                        # KV cache backfill: run remaining blocks so their
                        # KV caches are populated for future tokens.
                        for remaining_block in self.blocks[layer_idx + 1:]:
                            x = remaining_block(x)
                        return exit_logits, None, layer_idx, all_exit_logits
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
        return logits, loss, n_layer - 1, all_exit_logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss, _, _ = self(idx_cond)
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
    logits, _, _, _ = model(idx)
    
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
        logits, _, _, _ = model(idx_next, start_pos=idx.shape[1] - 1)
        
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
    loss = compute_joint_loss(model, xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# Generate: standard (full depth) vs early exit
# ---------------------------------------------------------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = 200

print("\n--- Full-depth generation (no early exit) ---")
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))

print("\n--- Early exit generation (threshold=0.9) ---")
clear_kv_cache(m)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
tokens = generate_early_exit(m, context, max_gen, exit_threshold=0.9)
print(decode(tokens[0].tolist()))


# ── non-cached generate (forces full-context recompute every step) ────────────
def generate_no_cache(model, idx, max_new_tokens):
    """Runs in train mode so the KV cache branch is never entered."""
    model.train()                          # disables KV cache path
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _, _ = model(idx_cond)
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
            logits, _, _, _ = model(idx[:, -1:])   # (B, 1, vocab_size)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx
    
