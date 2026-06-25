"""
NanoGPT + Tree-Based Speculative Decoding.

Extends chain speculative decoding (nanogpt-trigram-spec-decode.py) to a
branching candidate tree verified in a single forward pass via a custom tree
attention mask.

Builds on: nanogpt-trigram-spec-decode.py
Key additions:
    - TreeNode            — node in the speculation tree
    - draft_tree()        — depth-D, width-W tree from the bigram draft model
    - flatten_tree()      — DFS linearization + tree attention mask construction
    - verify_tree()       — single forward pass with tree attention mask
    - accept_reject_tree() — tree-walk rejection sampling, longest-path selection
    - trim_kv_cache_tree() — selective (non-contiguous) KV cache pruning
    - tree_speculative_generate() — main generation loop

See: notes/plans/tree-attention-spec-decode-plan.md

Run:
    python nanogpt-tree-attention.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import hashlib

# ---------------------------------------------------------------------------
# Hyperparameters (identical to nanogpt-trigram-spec-decode.py)
# ---------------------------------------------------------------------------

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

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
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
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Draft model (unchanged from nanogpt-trigram-spec-decode.py)
# ---------------------------------------------------------------------------

class BigramDraftModel:
    """P(next | current) from training data statistics."""

    def __init__(self, train_data, vocab_size, device):
        counts = torch.zeros(vocab_size, vocab_size, device=device)
        for i in range(len(train_data) - 1):
            counts[train_data[i], train_data[i + 1]] += 1
        counts += 1
        self.probs = counts / counts.sum(dim=1, keepdim=True)

    def get_probs(self, token_id):
        return self.probs[token_id]

    def sample(self, token_id):
        probs = self.get_probs(token_id)
        return torch.multinomial(probs, num_samples=1).item(), probs


# ---------------------------------------------------------------------------
# Phase 1 — Tree Data Structure
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A node in the speculation tree."""
    token_id: int
    draft_probs: Optional[torch.Tensor]   # (vocab_size,) — None for root
    parent: Optional['TreeNode']
    children: List['TreeNode']
    depth: int                             # 0 = root (current_token)
    linear_idx: int = -1                  # index in DFS-flattened list; -1 for root
    accepted: bool = False                 # filled by accept_reject_tree()
    resampled_token: Optional[int] = None  # filled on rejection

    @property
    def ancestors(self) -> List['TreeNode']:
        """Return the root-to-parent path (excluding self)."""
        path = []
        node = self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


def draft_tree(draft_model: BigramDraftModel, current_token: int,
               depth: int = 3, width: int = 2) -> TreeNode:
    """
    Build a speculation tree of depth `depth` and branching factor `width`.

    At each node, the top-`width` tokens from the draft distribution become
    children.  Returns the root TreeNode (token_id = current_token).

    TODO — implement this function.
    Hints:
      - Create the root with depth=0 and draft_probs=None.
      - Write a recursive `expand(node, remaining_depth)` helper that:
          1. Gets probs via draft_model.get_probs(node.token_id).
          2. Takes top-`width` tokens with torch.topk(probs, width).indices.
          3. Creates a child TreeNode for each and calls expand() on it.
      - Call expand(root, depth) then return root.
    See: plan §"Hint 1: Building the Tree"
    """

    root = TreeNode(token_id=current_token, draft_probs=None,
                    parent=None, children=[], depth=0)
    
    def expand(node: TreeNode, remaining_depth):
        if remaining_depth == 0:
            return
        
        probs = draft_model.get_probs(node.token_id)
        next_token_ids = torch.topk(probs, width).indices

        for token_id in next_token_ids:
            child = TreeNode(
                token_id=token_id,
                draft_probs=probs,
                parent=node,
                children=[],
                depth=node.depth + 1,
            )
            node.children.append(child)
            expand(child, remaining_depth - 1)

    expand(root, depth)
    return root

def flatten_tree(root: TreeNode):
    """
    Flatten the tree into DFS order (root excluded) and build the tree
    attention mask.

    Returns:
        nodes     : list[TreeNode]         — DFS order, root excluded
        tokens    : list[int]              — token_id for each node
        positions : list[int]              — depth offset for each node
                                             (add cache_len in verify_tree)
        mask      : (N, N) bool tensor     — tree attention mask
                                             mask[i, j] = True iff node i
                                             can attend to node j

    TODO — implement this function.
    Hints:
      - DFS: for each child, set child.linear_idx = len(nodes), append,
        then recurse.
      - Build mask as torch.zeros(N, N, dtype=torch.bool).
      - For each node i: set mask[i, i] = True (self-attention), then for
        each ancestor whose linear_idx >= 0, set mask[i, anc.linear_idx] = True.
      - Root (linear_idx == -1) is handled via past KV, not via the mask.
    See: plan §"Hint 2: Linearizing the Tree"
    """

    nodes = []

    def dfs(node):
        for child in node.children:
            child.linear_idx = len(nodes)
            nodes.append(child)
            dfs(child)

    dfs(root)
    root.linear_idx = -1  # root is implicit (already in KV cache or first input position)

    N = len(nodes)
    tokens = [n.token_id for n in nodes]
    positions = [n.depth for n in nodes]  # offset by cache_len later  

    # Build the tree attention mask
    # mask[i, j] = True iff node_i can attend to node_j
    mask = torch.zeros(N, N, dtype=torch.bool)
    for i, node in enumerate(nodes):
        # Each node attends to itself
        mask[i, i] = True
        # And to all its ancestors that are in the linearized list
        for anc in node.ancestors:
            if anc.linear_idx >= 0:  # skip root (handled via past KV)
                mask[i, anc.linear_idx] = True

    return nodes, tokens, positions, mask

    


# ---------------------------------------------------------------------------
# Phase 2 — Model with tree_attn_mask support
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """One head of self-attention with optional tree attention mask."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None, attn_mask=None,
                input_mask=None, tree_attn_mask=None):
        """
        Args:
            x:              (B, T, C)
            past_k:         (B, T_past, hs) or None
            past_v:         (B, T_past, hs) or None
            attn_mask:      (B, 1, T_past) — padding mask for cached positions
            input_mask:     (B, T)         — True = real token
            tree_attn_mask: (T, T) bool    — replaces the tril mask for the
                                             new-token region when provided;
                                             None = standard causal behaviour
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if not self.training:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

            T_full = k.shape[1]
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

            causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)

            if tree_attn_mask is not None:
                # tree_attn_mask is (1, T, T) — drop batch dim for indexing
                causal_mask[:, -T:] = tree_attn_mask[0]
            elif T > 1:
                new_token_mask = self.tril[:T, :T]
                causal_mask[:, -T:] = new_token_mask

            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

            if attn_mask is not None:
                if input_mask is not None:
                    new_valid = input_mask.unsqueeze(1)
                else:
                    new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
                full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)
                causal_mask = causal_mask & full_pad_mask

            wei = wei.masked_fill(~causal_mask, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, k, v
        else:
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, attn_mask=None, input_mask=None,
                tree_attn_mask=None):
        if past_kv is None:
            past_kv = [(None, None)] * len(self.heads)
        outputs, new_kvs = [], []
        for i, h in enumerate(self.heads):
            pk, pv = past_kv[i]
            out, nk, nv = h(x, pk, pv, attn_mask=attn_mask,
                            input_mask=input_mask,
                            tree_attn_mask=tree_attn_mask)
            outputs.append(out)
            new_kvs.append((nk, nv))
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_kvs


class FeedFoward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kv=None, attn_mask=None, input_mask=None,
                tree_attn_mask=None):
        sa_out, new_kv = self.sa(self.ln1(x), past_kv, attn_mask=attn_mask,
                                  input_mask=input_mask,
                                  tree_attn_mask=tree_attn_mask)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
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

    def forward(self, idx, targets=None, pos=None, past_kvs=None,
                attn_mask=None, input_mask=None, tree_attn_mask=None):
        """
        Args:
            tree_attn_mask: (1, T, T) bool — tree attention mask for the new-token
                            region, threaded through every Block → Head unchanged.
                            None = standard causal mask.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)

        if pos is None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        else:
            pos_emb = self.position_embedding_table(pos.clamp(max=block_size - 1))

        x = tok_emb + pos_emb

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, block_kv = block(x, past_kvs[i], attn_mask=attn_mask,
                                 input_mask=input_mask,
                                 tree_attn_mask=tree_attn_mask)
            new_kvs.append(block_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_2d = logits.view(B * T, C)
            targets_2d = targets.view(B * T)
            loss = F.cross_entropy(logits_2d, targets_2d)

        return logits, loss, new_kvs

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Training (identical to nanogpt-trigram-spec-decode.py)
# ---------------------------------------------------------------------------

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))


# ---------------------------------------------------------------------------
# Phase 3 — Tree Verification
# ---------------------------------------------------------------------------

def verify_tree(target_model: GPTLanguageModel, root: TreeNode,
                past_kvs, cache_len: int):
    """
    Verify all tree candidates in a single forward pass.

    Returns:
        target_probs : dict[TreeNode -> (vocab_size,) tensor]
        new_kvs      : updated KV cache
        nodes        : list[TreeNode] in DFS order (root excluded)

    TODO — implement this function.
    Hints:
      1. Call flatten_tree(root) to get nodes, tokens, positions, tree_mask.
      2. Prepend root: all_tokens = [root.token_id] + tokens.
      3. Offset positions by cache_len (root is depth 0, so pos 0+cache_len).
      4. Build full_mask (N+1, N+1):
           - full_mask[0, 0] = True           (root → self)
           - full_mask[1:, 0] = True           (all nodes → root)
           - full_mask[1:, 1:] = tree_mask     (tree structure)
      5. Call target_model(input_ids, pos=..., past_kvs=...,
                           tree_attn_mask=full_mask.unsqueeze(0)).
      6. Convert logits → softmax probabilities and store in a dict keyed by node.
    See: plan §"Hint 3: Tree Verification"
    """
    raise NotImplementedError("TODO: implement verify_tree()")


# ---------------------------------------------------------------------------
# Phase 4 — Tree Accept/Reject
# ---------------------------------------------------------------------------

def accept_reject_tree(root: TreeNode, nodes: List[TreeNode],
                       target_probs: dict) -> List[int]:
    """
    Walk the tree and apply rejection sampling at each node.

    Returns the token IDs along the best (longest) accepted root-to-leaf path,
    with a bonus token appended (sampled from the target distribution at the
    accepted leaf, just like chain spec decode).

    TODO — implement this function.
    Hints:
      1. Iterate over `nodes` in DFS order.  For each node:
           p = target_probs[node.parent][node.token_id]
           q = node.draft_probs[node.token_id]
           accept with probability min(p/q, 1).
         On rejection, compute residual distribution and store resampled_token.
      2. Find the longest accepted path with a recursive helper that only
         descends into children where child.accepted == True.
      3. If best_path is non-empty, append a bonus token sampled from
         target_probs[leaf_node].
      4. If best_path is empty (all first-level children rejected), fall back
         to the resampled token of the first child (or sample from target_probs[root]).
    See: plan §"Hint 4: Tree Accept/Reject"
    """
    raise NotImplementedError("TODO: implement accept_reject_tree()")


# ---------------------------------------------------------------------------
# Phase 5 — KV Cache Management
# ---------------------------------------------------------------------------

def trim_kv_cache_tree(new_kvs, accepted_path: List[int], root: TreeNode,
                       nodes: List[TreeNode], cache_len: int):
    """
    Trim the KV cache to keep only the accepted path's entries.

    After verify_tree() the cache contains:
        [past (cache_len tokens), root, node_0 ... node_{N-1}]

    We keep past + entries for nodes that lie on accepted_path (non-contiguous
    index gathering, unlike the simple slice used in chain spec decode).

    TODO — implement this function.
    Hints:
      1. Walk accepted_path (excluding the bonus token) to collect keep_indices.
         keep_indices = [0]  # root is at new-token index 0
         For each tok in accepted_path[:-1]:
             child = next child of current node where token_id==tok and accepted
             keep_indices.append(child.linear_idx + 1)  # +1 because root is 0
             current node = child
      2. For each layer and each (k, v) pair:
           past_k  = k[:, :cache_len, :]
           new_k   = k[:, cache_len:, :]
           selected_k = new_k[:, keep_indices, :]
           trimmed_k  = torch.cat([past_k, selected_k], dim=1)
         Same for v.
    See: plan §"Hint 5: Tree KV Cache Trimming"
    """
    raise NotImplementedError("TODO: implement trim_kv_cache_tree()")


# ---------------------------------------------------------------------------
# Phase 5 — Main generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def tree_speculative_generate(target_model: GPTLanguageModel,
                               draft_model: BigramDraftModel,
                               prompt_tokens: List[int],
                               max_new_tokens: int,
                               depth: int = 3,
                               width: int = 2) -> List[int]:
    """
    Full tree speculative decoding loop.

    TODO — wire together the five phases:
      1. Prefill: run target_model on prompt_tokens to get past_kvs and sample
         the first token.
      2. Loop until len(generated) >= max_new_tokens:
           a. draft_tree()          → root
           b. verify_tree()         → target_probs, new_kvs, nodes
           c. accept_reject_tree()  → accepted_path
           d. trim_kv_cache_tree()  → past_kvs
           e. extend generated, update current_token = accepted_path[-1]
      3. Return generated[:max_new_tokens].
    See: plan §"Putting It All Together"
    """
    raise NotImplementedError("TODO: implement tree_speculative_generate()")


# ---------------------------------------------------------------------------
# Quick smoke-test  (run after implementing all phases)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    draft_model = BigramDraftModel(train_data, vocab_size, device)

    prompt = encode("To be or not to be")
    print("\n--- Tree speculative generate (depth=3, width=2) ---")
    tokens = tree_speculative_generate(
        m, draft_model, prompt,
        max_new_tokens=200,
        depth=3,
        width=2,
    )
    print(decode(tokens))
