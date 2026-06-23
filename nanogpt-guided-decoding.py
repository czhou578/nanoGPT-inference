"""
NanoGPT + Guided Decoding — FSM-Constrained Generation.

Adds guided decoding to the KV-cached transformer. At each generation step,
a finite state machine (FSM) determines which tokens are allowed, and
disallowed tokens are masked to -inf before softmax. The model's forward
pass and KV cache are completely unchanged.

Builds on: nanogpt-kv-cache.py
Key additions:
    - build_char_classes(): maps 65-char vocabulary into reusable character classes
    - apply_token_mask(): masks logits so only allowed tokens can be sampled
    - GuidedFSM: state machine tracking position in a pattern
    - compile_pattern(): builds a GuidedFSM from a simple pattern string
    - generate_guided_static(): generation with pre-defined per-position masks
    - generate_guided(): FSM-guided generation with KV cache

Run:
    python nanogpt-guided-decoding.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

batch_size = 8
block_size = 64
max_iters = 120
eval_interval = 20
learning_rate = 1e-3
device = 'cpu'
eval_iters = 10
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
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
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Model (unchanged from nanogpt-kv-cache.py)
# ──────────────────────────────────────────────────────────────────────────────

def clear_kv_cache(model):
    for module in model.modules():
        if isinstance(module, Head):
            module.key_cache = None
            module.value_cache = None

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
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

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
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
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

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
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

    def forward(self, idx, targets=None, start_pos=0):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(start_pos, start_pos + T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ──────────────────────────────────────────────────────────────────────────────
# Unconstrained KV-cached generation (baseline, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def generate_kv_cache(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)

    # Prefill: process the initial context all at once
    logits, _ = model(idx)

    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        logits, _ = model(idx_next, start_pos=idx.shape[1] - 1)

    model.train()
    return idx

# ══════════════════════════════════════════════════════════════════════════════
#  GUIDED DECODING — everything below this line is new
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Character Classes
# ──────────────────────────────────────────────────────────────────────────────

def build_char_classes(stoi):
    """
    Build reusable character-class sets from the vocabulary.

    Each class maps to a set of token IDs. These sets are used to define
    FSM transitions and per-position masks.

    Returns a dict of class_name -> set of token IDs.

    Expected classes:
        UPPER, LOWER, LETTER, DIGIT, SPACE, NEWLINE, PUNCT, ANY
    """
    classes = {}

    for token in stoi:
        char = itos[token]
        if char.isupper():
            classes['UPPER'] = classes.get('UPPER', set()) | {token}
        elif char.islower():
            classes['LOWER'] = classes.get('LOWER', set()) | {token}
        elif char.isdigit():
            classes['DIGIT'] = classes.get('DIGIT', set()) | {token}
        elif char.isspace():
            classes['SPACE'] = classes.get('SPACE', set()) | {token}
        elif char == '\n':
            classes['NEWLINE'] = classes.get('NEWLINE', set()) | {token}
        else:
            classes['PUNCT'] = classes.get('PUNCT', set()) | {token}

    classes['LETTER'] = classes['UPPER'] | classes['LOWER']
    classes['ANY'] = set(stoi.values())

    return classes

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Token Masking Primitive
# ──────────────────────────────────────────────────────────────────────────────

def apply_token_mask(logits, allowed_token_ids):
    """
    Mask logits so only allowed tokens can be sampled.

    Args:
        logits:            (vocab_size,) raw logits from the model
        allowed_token_ids: set of token IDs that are permitted

    Returns:
        masked_logits:     (vocab_size,) with disallowed tokens set to -inf
    """

    if len(allowed_token_ids) == 0:
        raise ValueError("allowed_token_ids cannot be empty")

    mask = torch.zeros(vocab_size, dtype=bool, device=logits.device)
    mask[list(allowed_token_ids)] = True
    logits = logits.masked_fill(~mask, float('-inf'))
    return logits

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Static Per-Position Masks (Level 1)
# ──────────────────────────────────────────────────────────────────────────────

def generate_guided_static(model, idx, masks):
    """
    Generate with a pre-defined mask at each position.

    This is the simplest form of guided decoding: the mask at each step
    is fixed before generation starts (it doesn't depend on what was generated).

    Args:
        model:  GPTLanguageModel
        idx:    (1, T) prompt tensor
        masks:  list of sets, where masks[i] is the set of allowed
                token IDs at generation step i

    Returns:
        idx:    (1, T + len(masks)) the full sequence
    """
    model.eval()
    clear_kv_cache(model)

    logits, _ = model(idx)
    for i in range(len(masks)):
        logits = logits[:, -1, :]
        logits = apply_token_mask(logits, masks[i])
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        logits, _ = model(idx, start_pos=idx.shape[1] - 1)
    
    model.train()
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Finite State Machine (Level 2)
# ──────────────────────────────────────────────────────────────────────────────

class GuidedFSM:
    """
    Finite state machine for guided decoding.

    Each state maps to a list of transitions: (token_set, next_state_id).
    At each generation step, the current state determines which tokens
    are allowed. When a token is generated, the FSM transitions to the
    next state.

    Usage:
        fsm = GuidedFSM()
        # ... add states and transitions ...
        allowed = fsm.allowed_tokens()       # query what's valid
        masked = apply_token_mask(logits, allowed)
        token = sample(masked)
        fsm.advance(token)                   # advance state
    """
    def __init__(self):
        self.transitions = {}   # state_id -> list of (token_set, next_state_id)
        self.accept_states = set()
        self.current_state = 0

    def add_transition(self, from_state, token_set, to_state):
        """Add a transition: from_state --{token_set}--> to_state."""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append((token_set, to_state))

    def allowed_tokens(self):
        """
        Return the set of token IDs allowed in the current state.

        This is the union of all token sets from transitions leaving
        the current state.
        """

        allowed = set()

        if self.current_state not in self.transitions:
            return allowed 

        for ts, nt in self.transitions[self.current_state]:
            allowed.update(ts)

        return allowed

    def advance(self, token_id):
        """
        Advance the FSM by one token.

        Finds the transition whose token_set contains token_id,
        and moves to that transition's target state.

        Returns True if the transition was valid.
        """

        for ts, nt in self.transitions[self.current_state]:
            if token_id in ts:
                self.current_state = nt
                return True

        return False

        # TODO: implement this
        #
        # Iterate over transitions from self.current_state.
        # Find the first transition where token_id is in the token_set.
        # Update self.current_state to that transition's next_state.
        #
        # If no transition matches, return False (this means the mask
        # failed to prevent an invalid token — you have a bug).

        pass

    def is_complete(self):
        """Check if the FSM has reached an accept state."""
        return self.current_state in self.accept_states

    def reset(self):
        """Reset to the initial state for a new generation."""
        self.current_state = 0

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: FSM-Guided Generation (Level 2)
# ──────────────────────────────────────────────────────────────────────────────

def generate_guided(model, idx, fsm, max_new_tokens):
    """
    Generate with FSM-guided decoding using KV cache.

    At each step:
      1. Get logits from the model
      2. Query the FSM for allowed tokens
      3. Mask logits with apply_token_mask
      4. Sample a token
      5. Advance the FSM
      6. Stop if FSM reaches an accept state

    Args:
        model:          GPTLanguageModel (in eval mode for KV cache)
        idx:            (1, T) prompt tensor
        fsm:            GuidedFSM instance (will be mutated)
        max_new_tokens: safety cap to prevent infinite generation

    Returns:
        idx: (1, T + generated) the full sequence
    """
    model.eval()
    clear_kv_cache(model)
    fsm.reset()

    logits, _ = model(idx)
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]
        allowed = fsm.allowed_tokens()
        logits = apply_token_mask(logits, allowed)
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        fsm.advance(idx_next.item())

        if fsm.is_complete():
            break
        
        logits, _ = model(idx, start_pos=idx.shape[1] - 1)
    
    model.train()
    return idx

    # TODO: implement this
    #
    # Start from generate_kv_cache() and add 3 lines:
    #
    #   logits, _ = model(idx)                              # prefill
    #   for _ in range(max_new_tokens):
    #       logits = logits[:, -1, :]
    #       allowed = fsm.allowed_tokens()                  # ← NEW
    #       logits = apply_token_mask(logits, allowed)      # ← NEW
    #       probs = F.softmax(logits, dim=-1)
    #       idx_next = torch.multinomial(probs, num_samples=1)
    #       idx = torch.cat((idx, idx_next), dim=1)
    #       fsm.advance(idx_next.item())                    # ← NEW
    #       if fsm.is_complete():                           # ← NEW
    #           break
    #       logits, _ = model(idx_next, start_pos=idx.shape[1] - 1)
    #
    # Don't forget model.train() at the end.

    pass

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Pattern Compiler (Level 2 bonus)
# ──────────────────────────────────────────────────────────────────────────────

def compile_pattern(pattern_elements, char_classes, stoi):
    """
    Compile a simple pattern into a GuidedFSM.

    Pattern elements are a list of (class_or_literal, quantifier) tuples.
    Supported quantifiers: '1' (exactly one), '+' (one or more).

    Example - the pattern UPPER+: LOWER+\\n becomes:
        [
            ('UPPER', '+'),     # one or more uppercase letters
            (':',     '1'),     # literal colon
            (' ',     '1'),     # literal space
            ('LOWER', '+'),     # one or more lowercase letters
            ('\\n',    '1'),     # literal newline
        ]

    Args:
        pattern_elements: list of (class_name_or_char, quantifier) tuples
        char_classes:     dict from build_char_classes()
        stoi:             character-to-token-id mapping

    Returns:
        GuidedFSM ready for use with generate_guided()
    """
    fsm = GuidedFSM()
    state_id = 0

    for class_or_char, quantifier in pattern_elements:
        # Resolve the token set: either a named character class or a literal char
        if class_or_char in char_classes:
            token_set = char_classes[class_or_char]
        else:
            token_set = {stoi[class_or_char]}

        if quantifier == '1':
            # Exactly one: state_id --{tokens}--> state_id+1
            fsm.add_transition(state_id, token_set, state_id + 1)
            state_id += 1

        elif quantifier == '+':
            # One or more:
            #   state_id   --{tokens}--> state_id+1   (must match at least one)
            #   state_id+1 --{tokens}--> state_id+1   (self-loop: more is okay)
            fsm.add_transition(state_id, token_set, state_id + 1)
            fsm.add_transition(state_id + 1, token_set, state_id + 1)
            state_id += 1

    fsm.accept_states = {state_id}
    return fsm

# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ── Unconstrained baseline ───────────────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = block_size - context.shape[1]
print("\n── Unconstrained generation ──")
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))

# ══════════════════════════════════════════════════════════════════════════════
#  Tests — uncomment these as you implement each step
# ══════════════════════════════════════════════════════════════════════════════

# ── Test: build_char_classes ─────────────────────────────────────────────────
# char_classes = build_char_classes(stoi)
# print(f"\nCharacter classes:")
# for name, ids in sorted(char_classes.items()):
#     chars_in_class = [itos[i] for i in sorted(ids)]
#     print(f"  {name:8s} ({len(ids):2d} tokens): {''.join(chars_in_class)}")

# ── Test Level 1: static masks (5 lowercase + newline) ──────────────────────
# char_classes = build_char_classes(stoi)
# lowercase_ids = char_classes['LOWER']
# newline_ids = char_classes['NEWLINE']
#
# masks = [lowercase_ids] * 5 + [newline_ids]
# prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
# output = generate_guided_static(m, prompt, masks)
# generated = decode(output[0].tolist())
# print(f"\n── Static mask test (5 lowercase + newline) ──")
# print(f"Generated: {repr(generated)}")
# # Verify: last 6 chars should be [a-z]{5}\n
# suffix = generated[-6:]
# assert suffix[-1] == '\n', f"Expected newline at end, got {repr(suffix[-1])}"
# assert all(c.islower() for c in suffix[:-1]), f"Expected lowercase, got {repr(suffix[:-1])}"
# print("✓ Constraint satisfied!")

# ── Test Level 2: manual FSM (UPPER+: LOWER+\n) ─────────────────────────────
# char_classes = build_char_classes(stoi)
#
# fsm = GuidedFSM()
# fsm.add_transition(0, char_classes['UPPER'], 1)   # must see >= 1 uppercase
# fsm.add_transition(1, char_classes['UPPER'], 1)   # self-loop: more uppercase
# fsm.add_transition(1, {stoi[':']}, 2)             # colon
# fsm.add_transition(2, {stoi[' ']}, 3)             # space
# fsm.add_transition(3, char_classes['LOWER'], 4)   # must see >= 1 lowercase
# fsm.add_transition(4, char_classes['LOWER'], 4)   # self-loop: more lowercase
# fsm.add_transition(4, {stoi['\n']}, 5)            # newline
# fsm.accept_states = {5}
#
# prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
# output = generate_guided(m, prompt, fsm, max_new_tokens=30)
# generated = decode(output[0].tolist())
# print(f"\n── FSM test (UPPER+: LOWER+\\n) ──")
# print(f"Generated: {repr(generated)}")
#
# # Run it 5 times to see variety
# for i in range(5):
#     torch.manual_seed(1337 + i)
#     fsm.reset()
#     clear_kv_cache(m)
#     output = generate_guided(m, prompt, fsm, max_new_tokens=30)
#     line = decode(output[0].tolist())
#     # strip prompt (first char)
#     text = line[1:]  # skip the \0 prompt token
#     print(f"  Run {i}: {repr(text)}")

# ── Test Level 2 bonus: compile_pattern ──────────────────────────────────────
# char_classes = build_char_classes(stoi)
#
# pattern = [
#     ('UPPER', '+'),
#     (':',     '1'),
#     (' ',     '1'),
#     ('LOWER', '+'),
#     ('\n',    '1'),
# ]
# fsm = compile_pattern(pattern, char_classes, stoi)
#
# prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
# output = generate_guided(m, prompt, fsm, max_new_tokens=30)
# generated = decode(output[0].tolist())
# print(f"\n── compile_pattern test ──")
# print(f"Generated: {repr(generated)}")
