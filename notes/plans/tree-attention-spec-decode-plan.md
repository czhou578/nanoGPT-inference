# Tree-Based Speculative Decoding — Implementation Plan

## The Problem You're Solving

Your existing speculative decoding drafts a **single chain** of K tokens and verifies them sequentially. If the first rejection happens at position 2, you lose positions 3 and 4 entirely — their compute is wasted.

```
Chain speculation (current):
  token_0 → token_1 → token_2 → token_3 → token_4
                                    ✗ rejected
                        (tokens 3,4 wasted)
```

Tree-based speculative decoding drafts a **branching tree** of candidates. At each position, the draft model proposes multiple continuations. All branches are verified in a **single forward pass** using a custom tree attention mask. When one branch is rejected, other branches at the same depth may still be accepted.

```
Tree speculation (this plan):
  token_0 → token_1a → token_2a → token_3a
                     → token_2b
          → token_1b → token_2c
                     → token_2d

  6 candidates verified in 1 forward pass instead of 4.
  If token_2a is rejected, token_2b may still be accepted.
```

The key insight: the transformer's attention mechanism already supports arbitrary causal dependency patterns through the attention mask. A standard causal mask encodes a linear chain. A **tree attention mask** encodes a tree — each node attends only to its ancestors, not to its siblings or cousins.

---

## What You Already Have

From `nanogpt-trigram-spec-decode.py`:

- ✅ `BigramDraftModel` and `TrigramDraftModel` — cheap next-token predictors
- ✅ `draft_tokens()` — generates a chain of K candidates
- ✅ `verify_candidates()` — single forward pass verification with KV cache
- ✅ `accept_reject()` — rejection sampling with residual resampling
- ✅ `trim_kv_cache()` — cache rollback on rejection
- ✅ `Head.forward()` already accepts a custom `attn_mask` parameter
- ✅ Full PagedAttention + scheduling + prefix caching integration

What's missing: tree construction, tree attention mask generation, tree-aware verification, and tree-aware accept/reject with path selection.

---

## The Tree Attention Mask — The Core Idea

### Standard causal mask (linear chain)

When verifying `[cur, c0, c1, c2, c3]` (your current approach), the causal mask looks like:

```
        past  cur  c0   c1   c2   c3
cur  [  1     1    0    0    0    0  ]
c0   [  1     1    1    0    0    0  ]
c1   [  1     1    1    1    0    0  ]
c2   [  1     1    1    1    1    0  ]
c3   [  1     1    1    1    1    1  ]
```

Every token can attend to all previous tokens. This encodes a single linear sequence.

### Tree attention mask (branching)

Suppose the draft tree looks like:

```
cur → c0 → c2
         → c3
    → c1 → c4
         → c5
```

The tree is linearized (DFS order) to `[cur, c0, c2, c3, c1, c4, c5]`.
The tree attention mask becomes:

```
        past  cur  c0   c2   c3   c1   c4   c5
cur  [  1     1    0    0    0    0    0    0  ]
c0   [  1     1    1    0    0    0    0    0  ]
c2   [  1     1    1    1    0    0    0    0  ]   ← c2 attends to cur,c0 (its ancestors)
c3   [  1     1    1    0    1    0    0    0  ]   ← c3 attends to cur,c0 (NOT c2 — sibling)
c1   [  1     1    0    0    0    1    0    0  ]   ← c1 attends to cur only (NOT c0 — sibling)
c4   [  1     1    0    0    0    1    1    0  ]   ← c4 attends to cur,c1
c5   [  1     1    0    0    0    1    0    1  ]   ← c5 attends to cur,c1 (NOT c4)
```

Each token attends to its **direct ancestors** in the tree, not to siblings or cousins. This is the only change needed to enable tree verification — the transformer math is identical.

### Position indices

Each node's position index equals `cache_len + depth_in_tree`, not its index in the linearized sequence. Siblings share the same position index because they represent alternative tokens at the same sequence position.

```
Node:    cur   c0   c2   c3   c1   c4   c5
Depth:    0    1    2    2    1    2    2
Pos:      L   L+1  L+2  L+2  L+1  L+2  L+2
```

This ensures positional embeddings correctly reflect each token's distance from the prompt.

---

## The Tree Data Structure

```python
@dataclass
class TreeNode:
    """A node in the speculation tree."""
    token_id: int
    draft_probs: torch.Tensor      # (vocab_size,) — draft distribution at this node
    parent: 'TreeNode | None'
    children: list['TreeNode']
    depth: int                      # 0 = root (current_token)
    linear_idx: int = -1           # index in the linearized sequence (set during flatten)

    @property
    def ancestors(self) -> list['TreeNode']:
        """Return root-to-parent path (excluding self)."""
        path = []
        node = self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
```

The root node is `current_token` (depth 0). Its children are the first-level draft candidates. Each child can itself have children (second-level candidates), and so on.

---

## Hint 1: Building the Tree — `draft_tree()`

Extend `draft_tokens()` from a chain to a tree. At each depth level, sample `W` (width) candidates per parent. The tree has depth `D` and width `W`, producing up to `W^D` leaf nodes (but typically pruned).

```python
def draft_tree(draft_model, current_token, depth=3, width=2):
    """
    Build a speculation tree of candidate tokens.

    Args:
        draft_model:    BigramDraftModel or TrigramDraftModel
        current_token:  the last accepted token
        depth:          maximum tree depth (D)
        width:          candidates per node (W)

    Returns:
        root: TreeNode — the root of the speculation tree
    """
    root = TreeNode(
        token_id=current_token,
        draft_probs=None,      # root has no draft distribution
        parent=None,
        children=[],
        depth=0,
    )

    def expand(node, remaining_depth):
        if remaining_depth == 0:
            return
        probs = draft_model.get_probs(node.token_id)
        # Sample W distinct children (top-W from the distribution)
        top_tokens = torch.topk(probs, width).indices
        for tok in top_tokens:
            child = TreeNode(
                token_id=tok.item(),
                draft_probs=probs,
                parent=node,
                children=[],
                depth=node.depth + 1,
            )
            node.children.append(child)
            expand(child, remaining_depth - 1)

    expand(root, depth)
    return root
```

**Why top-W instead of sampling?** Top-W gives the most likely branches, maximizing acceptance probability. With a bigram draft, the top-2 continuations already cover most of the probability mass. Sampling introduces diversity but wastes tree budget on low-probability branches.

**Tree budget:** A depth-3, width-2 tree has 2 + 4 + 8 = 14 nodes (excluding root). That's 14 candidates verified in one forward pass. Compare with the chain approach: 4 candidates in one pass. The tree explores more of the probability space at the cost of a larger (but still single) forward pass.

---

## Hint 2: Linearizing the Tree — `flatten_tree()`

The transformer expects a flat `(1, T)` input tensor. The tree must be serialized into a sequence. DFS order is natural and keeps ancestors contiguous.

```python
def flatten_tree(root):
    """
    Flatten the tree into a list of nodes (DFS order, excluding root).

    Returns:
        nodes:     list of TreeNode (in DFS order, root excluded)
        tokens:    list of int — token IDs
        positions: list of int — position offsets (depth-based)
        mask:      (N, N) bool tensor — tree attention mask for the N nodes
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
```

The mask is a sparse boolean matrix — much sparser than the standard lower-triangular causal mask. This sparsity is what allows sibling branches to be independent.

---

## Hint 3: Tree Verification — `verify_tree()`

Replace `verify_candidates()` with a tree-aware version. The key change: pass the tree attention mask to the model instead of the standard causal mask.

```python
def verify_tree(target_model, root, past_kvs, cache_len):
    """
    Verify all tree candidates in a single forward pass.

    Args:
        target_model: GPTLanguageModel
        root:         TreeNode (the speculation tree root)
        past_kvs:     KV cache from previous steps
        cache_len:    number of tokens already in the KV cache

    Returns:
        target_probs: dict mapping TreeNode → (vocab_size,) target distribution
        new_kvs:      updated KV cache
    """
    nodes, tokens, positions, tree_mask = flatten_tree(root)

    # Prepend root token (current_token)
    all_tokens = [root.token_id] + tokens
    all_positions = [0] + positions  # root is at depth 0

    # Offset positions by cache_len
    all_positions = [cache_len + p for p in all_positions]

    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)
    pos = torch.tensor([all_positions], device=device)

    # Build full attention mask: (1, N+1, N+1)
    # Row 0 (root) attends to itself only (among new tokens)
    # Rows 1..N use the tree_mask, shifted by 1, plus attending to root
    N = len(nodes)
    full_mask = torch.zeros(N + 1, N + 1, dtype=torch.bool)
    full_mask[0, 0] = True                       # root attends to self
    full_mask[1:, 0] = True                      # all nodes attend to root
    full_mask[1:, 1:] = tree_mask                # tree structure

    # The model's Head.forward already handles past KV attention via attn_mask.
    # We pass full_mask as a custom causal mask for the NEW tokens only.
    # This replaces the standard tril mask in Head.forward.

    logits, _, new_kvs = target_model(
        input_ids, pos=pos, past_kvs=past_kvs,
        tree_attn_mask=full_mask.unsqueeze(0),  # (1, N+1, N+1)
    )

    # Extract per-node target probabilities
    target_probs = {}
    target_probs[root] = F.softmax(logits[0, 0, :], dim=-1)
    for i, node in enumerate(nodes):
        target_probs[node] = F.softmax(logits[0, i + 1, :], dim=-1)

    return target_probs, new_kvs, nodes
```

### Required model change

`Head.forward()` currently builds a causal mask internally. For tree attention, we need to pass an explicit mask for the new-token region. The change is small — see "What Changes in the Model" below.

---

## Hint 4: Tree Accept/Reject — `accept_reject_tree()`

The acceptance logic walks the tree top-down. At each node, apply the same rejection sampling as before. The key difference: when a node is rejected, **prune its entire subtree**, but continue checking its siblings.

After processing all nodes, select the **longest accepted path** from root to a leaf.

```python
def accept_reject_tree(root, nodes, target_probs):
    """
    Walk the tree and apply rejection sampling at each node.

    Returns:
        accepted_path: list of token IDs along the best accepted path
        accepted_depth: how deep the acceptance went
    """
    # Mark each node as accepted or rejected
    for node in nodes:
        parent = node.parent
        # parent's target distribution predicts what should come at this position
        p = target_probs[parent][node.token_id]
        q = node.draft_probs[node.token_id]

        ratio = (p / q).clamp(max=1.0).item()
        node.accepted = torch.rand(1).item() < ratio

        if not node.accepted:
            # Resample from residual distribution
            adjusted = torch.clamp(target_probs[parent] - node.draft_probs, min=0)
            adj_sum = adjusted.sum()
            if adj_sum > 0:
                adjusted = adjusted / adj_sum
            else:
                adjusted = target_probs[parent]
            node.resampled_token = torch.multinomial(adjusted, num_samples=1).item()

    # Find the longest fully-accepted root-to-leaf path
    best_path = []

    def find_best(node, current_path):
        nonlocal best_path
        for child in node.children:
            if child.accepted:
                child_path = current_path + [child.token_id]
                if len(child_path) > len(best_path):
                    best_path = child_path
                find_best(child, child_path)

    find_best(root, [])

    # If we have a fully accepted path, sample a bonus token from the last node's
    # target distribution (same as chain spec decode)
    if best_path:
        # Find the leaf node of the best path
        node = root
        for tok in best_path:
            node = next(c for c in node.children if c.token_id == tok and c.accepted)
        bonus = torch.multinomial(target_probs[node], num_samples=1).item()
        best_path.append(bonus)
    else:
        # All first-level children rejected — use the resampled token from
        # the first child (or sample from target)
        first_child = nodes[0]  # first in DFS order
        if hasattr(first_child, 'resampled_token'):
            best_path = [first_child.resampled_token]
        else:
            best_path = [torch.multinomial(target_probs[root], num_samples=1).item()]

    return best_path
```

**Why longest path?** The goal is to maximize tokens per forward pass. If branch A accepts 3 tokens and branch B accepts 1, we take branch A. The tokens on branch B are discarded (their KV entries will be trimmed).

---

## Hint 5: Tree KV Cache Trimming

After selecting the accepted path, the KV cache contains entries for **all** tree nodes. We need to keep only the entries corresponding to the accepted path.

```python
def trim_kv_cache_tree(new_kvs, accepted_path, root, nodes, cache_len):
    """
    Trim KV cache to keep only the accepted path's entries.

    The cache after verification contains entries for:
      [past_kvs..., root, node_0, node_1, ..., node_{N-1}]

    We need to keep past_kvs + the entries for nodes on the accepted path.
    """
    # Find which linear indices correspond to the accepted path
    keep_indices = [0]  # root (index 0 in new tokens)
    node = root
    for tok in accepted_path[:-1]:  # exclude bonus token (not in cache)
        child = next(c for c in node.children if c.token_id == tok and c.accepted)
        keep_indices.append(child.linear_idx + 1)  # +1 because root is at 0
        node = child

    # Rearrange KV cache: keep past + selected entries in order
    trimmed = []
    for layer_kv in new_kvs:
        layer_trimmed = []
        for (k, v) in layer_kv:
            # k shape: (1, cache_len + N + 1, head_size)
            past_k = k[:, :cache_len, :]
            new_k = k[:, cache_len:, :]
            selected_k = new_k[:, keep_indices, :]
            trimmed_k = torch.cat([past_k, selected_k], dim=1)

            past_v = v[:, :cache_len, :]
            new_v = v[:, cache_len:, :]
            selected_v = new_v[:, keep_indices, :]
            trimmed_v = torch.cat([past_v, selected_v], dim=1)

            layer_trimmed.append((trimmed_k, trimmed_v))
        trimmed.append(layer_trimmed)

    return trimmed
```

This is more complex than chain trimming (which just slices `[:keep]`) because the accepted entries are not necessarily contiguous in the linearized tree.

---

## What Changes in the Model

The only model change is in `Head.forward()`: accept an optional `tree_attn_mask` that replaces the internally-constructed causal mask for the new-token region.

```python
# In Head.forward(), replace the causal mask construction:

# BEFORE (standard causal):
causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
if T > 1:
    new_token_mask = self.tril[:T, :T]
    causal_mask[:, -T:] = new_token_mask

# AFTER (tree-aware):
causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
if tree_attn_mask is not None:
    # Use the provided tree mask for new-token region
    causal_mask[:, -T:] = tree_attn_mask  # (T, T)
elif T > 1:
    new_token_mask = self.tril[:T, :T]
    causal_mask[:, -T:] = new_token_mask
```

When `tree_attn_mask` is `None`, behavior is identical to before. This keeps the change non-invasive — all existing code paths work unchanged.

The `tree_attn_mask` parameter needs to be threaded through `MultiHeadAttention.forward()`, `Block.forward()`, and `GPTLanguageModel.forward()`, but this is purely plumbing — add `tree_attn_mask=None` to each signature and pass it through.

---

## Putting It All Together — `tree_speculative_generate()`

```python
@torch.no_grad()
def tree_speculative_generate(target_model, draft_model, prompt_tokens,
                               max_new_tokens, depth=3, width=2):
    target_model.eval()
    generated = []

    # 1. Prefill
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
    logits, _, past_kvs = target_model(input_ids, pos=positions)

    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(current_token)

    # 2. Tree speculative decode loop
    while len(generated) < max_new_tokens:
        cache_len = past_kvs[0][0][0].shape[1]

        # DRAFT TREE
        root = draft_tree(draft_model, current_token, depth=depth, width=width)

        # VERIFY (single forward pass)
        target_probs, new_kvs, nodes = verify_tree(
            target_model, root, past_kvs, cache_len
        )

        # ACCEPT/REJECT (tree walk)
        accepted_path = accept_reject_tree(root, nodes, target_probs)

        # TRIM KV CACHE (keep only accepted path)
        past_kvs = trim_kv_cache_tree(new_kvs, accepted_path, root, nodes, cache_len)

        # Update state
        generated.extend(accepted_path)
        current_token = accepted_path[-1]

    return generated[:max_new_tokens]
```

---

## Recommended Build Order

```
Phase 1: Tree Data Structure
  1. TreeNode dataclass
  2. draft_tree() — build a depth-D, width-W tree from the bigram model
  3. flatten_tree() — DFS linearization + tree attention mask generation
  4. Tests: verify mask shape, verify ancestors are marked correctly

Phase 2: Tree Attention Mask in the Model
  5. Add tree_attn_mask parameter to Head.forward()
  6. Thread through MultiHeadAttention → Block → GPTLanguageModel
  7. Test: verify that tree_attn_mask=None produces identical output to before
  8. Test: verify a tree mask with only diagonal entries produces independent predictions

Phase 3: Tree Verification
  9. verify_tree() — single forward pass with tree mask
  10. Test: verify that a depth-1 tree (chain) produces same probs as verify_candidates()

Phase 4: Tree Accept/Reject
  11. accept_reject_tree() — walk tree, apply rejection sampling, find best path
  12. Test: if draft_model == target_model (probabilities match), acceptance rate ≈ 100%
  13. Test: single-chain tree (width=1) matches existing accept_reject() behavior

Phase 5: KV Cache Management
  14. trim_kv_cache_tree() — selective index gathering
  15. tree_speculative_generate() — main loop
  16. Test: greedy equivalence — greedy tree spec decode == greedy standard decode

Phase 6: Benchmarks
  17. Compare acceptance rate: chain K=4 vs tree D=3,W=2
  18. Compare tokens per forward pass
  19. Compare total forward passes for a fixed generation length
  20. Blog post with results
```

---

## Key Metrics to Track

| Metric | Chain (current) | Tree (expected) |
|--------|----------------|-----------------|
| Candidates per verify call | K (e.g., 4) | W+W²+...+W^D (e.g., 14 for D=3,W=2) |
| Tokens per verify call | 1–5 | 1–D+1 (more likely to hit max) |
| Acceptance rate per token | ~30-40% (bigram) | Similar per-token, but more chances |
| Effective tokens/forward pass | ~1.3-1.6 | ~1.5-2.5 (from branch diversity) |
| Forward pass cost | O(K) | O(W^D) — larger but still single pass |

The tree wins when the **total accepted tokens per forward pass** exceeds the chain despite the higher per-pass cost. The sweet spot is typically D=2-3, W=2-3.

---

## Gotchas

1. **Position collisions.** Siblings share the same position index. The position embedding table handles this fine (it's just a lookup), but be careful not to assume positions are unique within the input tensor.

2. **Tree mask shape vs cache mask.** The tree attention mask only governs attention among the *new* tokens. Attention to *past* KV entries (the cache) is always allowed (all 1s), same as before. Don't confuse the two mask regions.

3. **Bonus token on the best path only.** The bonus token (sampled from the target distribution at the accepted leaf) is only generated for the selected path. Other paths' target distributions are discarded.

4. **Tree budget vs. forward pass cost.** A depth-4, width-3 tree has 120 nodes. The forward pass cost scales with the total number of nodes. Keep the tree small enough that the single forward pass is cheaper than multiple chain passes.

5. **The `block_size=64` position limit still applies.** Total sequence length (prompt + generated + max tree depth) must fit within the position embedding table. With tree depth D, each forward pass consumes at most D new position slots.

6. **Resampled tokens on rejected branches.** When a node is rejected and resampled, its children (computed with the original drafted token) are invalid. The subtree must be pruned — you cannot use a resampled parent with children that were drafted from the original token.

---

## File Structure

```
nanogpt-tree-attention.py          # Main implementation
  - TreeNode                        # Tree data structure
  - draft_tree()                    # Build speculation tree from draft model
  - flatten_tree()                  # DFS linearization + mask construction
  - verify_tree()                   # Single forward pass with tree mask
  - accept_reject_tree()            # Tree-aware rejection sampling
  - trim_kv_cache_tree()            # Selective KV cache pruning
  - tree_speculative_generate()     # Main generation loop

  # Model changes (minimal):
  - Head.forward()                  # +tree_attn_mask parameter
  - MultiHeadAttention.forward()    # passthrough
  - Block.forward()                 # passthrough
  - GPTLanguageModel.forward()      # passthrough

  # Everything else copied from nanogpt-trigram-spec-decode.py:
  - BigramDraftModel, TrigramDraftModel
  - PagedAttention, BlockAllocator, Scheduler, etc.
```

---

## What Makes This Educational

Most tree spec decode tutorials focus on Medusa (which requires training extra prediction heads) or SpecInfer (which uses a separate small draft transformer). Your approach is simpler and more instructive:

1. **No extra model parameters.** You reuse the existing bigram/trigram tables. The tree structure comes purely from sampling multiple candidates at each step.

2. **The tree attention mask is the entire contribution.** Everything else — the model, the draft, the rejection sampling math — is identical to chain spec decode. This isolates the concept cleanly.

3. **It shows why attention masks are powerful.** The same transformer, with only a different boolean mask, can verify an exponentially larger space of candidates. Students see that the "causal" in causal attention is a policy choice (the mask), not a structural constraint.
