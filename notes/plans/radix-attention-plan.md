# Radix Attention — Implementation Plan & Hints

## The Problem You're Solving

Your current prefix caching in `nanogpt-prefix-caching.py` uses a **flat hash map** (`BlockCache`). Every block is identified by a chained MD5 hash of `(parent_hash, token_ids)`, and the cache is a flat `Dict[bytes, CachedBlock]`. This works — but it has structural limitations that SGLang's RadixAttention solves.

### What's wrong with flat prefix caching?

Consider this multi-turn chat workload:

```
Turn 1:  [system_prompt] + [user_msg_1]           → generates response_1
Turn 2:  [system_prompt] + [user_msg_1] + [response_1] + [user_msg_2]  → generates response_2
Turn 3a: [system_prompt] + [user_msg_1] + [response_1] + [user_msg_3]  → generates response_3a (branch!)
Turn 3b: [system_prompt] + [user_msg_1] + [response_1] + [user_msg_4]  → generates response_3b (branch!)
```

With a flat cache, Turn 3a and Turn 3b **independently** look up blocks from position 0, walking the chain sequentially. There's no structural awareness that they share the same path through `[system_prompt] + [user_msg_1] + [response_1]`. Worse:

1. **No branching visibility.** You can't ask "what are all the suffixes that extend this prefix?" The flat map can't answer structural queries.
2. **Eviction is blind.** LRU eviction picks the globally oldest block. But if Block 5 of a shared prefix is evicted, then Blocks 6, 7, 8 become unreachable — their chained hashes depend on Block 5 existing. The flat cache doesn't know about these dependencies, so it might evict a critical interior node while keeping useless orphaned leaf blocks.
3. **Redundant traversal.** Every `find_cached_prefix()` call re-walks from block 0, re-hashing and re-looking-up every block. With a tree, you traverse edges and the shared prefix is implicit.

### What RadixAttention does differently

SGLang replaces the flat hash map with a **radix tree** (compressed trie). Token sequences are stored as paths from root to leaf. Shared prefixes are shared edges — stored once, referenced by all requests that pass through them. This gives you:

- **O(1) fork detection**: branching points are tree nodes with multiple children
- **Structural eviction**: evict leaves first, never orphan interior nodes
- **Longest-prefix matching**: traverse the tree once, no re-hashing needed
- **Memory sharing**: shared prefixes are stored once in the tree, not duplicated

---

## Why This Matters Even at 210K Params

Same as with prefix caching — you won't see wall-clock gains on nanoGPT. But you're learning the exact data structure that SGLang uses in production. The concepts transfer directly:

1. **Radix tree with compressed edges** — each node stores a variable-length sequence of tokens, not just one token
2. **Node splitting** — when a new sequence partially matches an existing edge, the edge is split to create a branch point
3. **Reference counting (lock_ref)** — active requests "pin" their path in the tree to prevent eviction
4. **Leaf-first eviction** — only evict unpinned leaf nodes; their parents become new leaves if also unpinned

---

## How the Current Code Maps to the New Code

Here's exactly what you're replacing and what stays the same:

| Current (flat) | New (radix tree) | What changes |
|---|---|---|
| `hash_block_tokens()` | **Deleted** — tree structure encodes prefix chains implicitly | No more explicit hashing |
| `NONE_HASH` sentinel | **Deleted** — the root node is the implicit "no parent" | — |
| `CachedBlock` dataclass | `RadixNode` class | Node stores children, parent pointer, KV data, and token slice |
| `BlockCache` (flat dict) | `RadixTree` class | Tree with `match_prefix()`, `insert()`, `evict_lru()` |
| `find_cached_prefix()` | `RadixTree.match_prefix()` | Tree traversal instead of hash-chain walk |
| `load_cached_blocks()` | `load_from_radix_tree()` | Collect KV data along matched path |
| `commit_completed_blocks()` | `insert_into_radix_tree()` | Insert new KV blocks as tree nodes |
| `Scheduler`, `Request`, `Head`, model | **Unchanged** | The model still receives `past_kvs` — it can't tell the difference |

---

## Hint 1: The RadixNode Data Structure

Each node in the tree stores a **variable-length** slice of tokens (not a fixed block). This is the "compressed" part of the compressed trie — if a sequence of tokens has no branches, it's stored as a single edge.

```python
class RadixNode:
    def __init__(self):
        self.children: Dict[int, RadixNode] = {}  # first token of child → child node
        self.parent: Optional[RadixNode] = None
        self.token_ids: Tuple[int, ...] = ()      # the token sequence this edge represents
        self.kv_data: Optional[Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]]] = None
        self.lock_ref: int = 0                     # number of active requests using this node
        self.last_access_time: int = 0             # for LRU eviction
```

**Key difference from `CachedBlock`:** a `RadixNode` has `children` and `parent`. It knows its place in the tree. When you evict it, you can walk up to its parent and check if the parent becomes a leaf.

**The `children` dict key is the first token of the child's `token_ids`.** This lets you do O(1) lookup when traversing: given the next token in a query, you check `node.children.get(next_token)`. This is the radix tree's branching mechanism.

**Question to think about:** Why is the dict key just the *first* token, not the whole `token_ids` tuple?

Answer: Because you need to check whether any child *starts with* the next token. A child edge might be `(10, 20, 30, 40)` — you match the first token `10` to find the right child, then compare the rest of the edge against the query. If they diverge mid-edge, you split. If the key were the full tuple, you'd need to check every child's first token — defeating the purpose.

---

## Hint 2: Match Prefix — Tree Traversal

This is the core algorithm. Given a query token sequence, walk the tree from the root, consuming tokens as you match edges. Stop when you run out of matching edges or query tokens.

```python
def match_prefix(self, token_ids: List[int]) -> Tuple[RadixNode, int]:
    """
    Find the longest prefix of token_ids that exists in the tree.
    Returns (last_matched_node, num_matched_tokens).
    
    IMPORTANT: If the match ends in the MIDDLE of an edge, you must
    SPLIT the edge so there's a node at the exact match boundary.
    """
    node = self.root
    matched = 0
    
    while matched < len(token_ids):
        next_token = token_ids[matched]
        child = node.children.get(next_token)
        if child is None:
            break  # no edge starts with this token — match ends here
        
        # Compare the query against this edge's tokens
        edge_tokens = child.token_ids
        edge_match_len = 0
        while (edge_match_len < len(edge_tokens) and 
               matched + edge_match_len < len(token_ids) and
               edge_tokens[edge_match_len] == token_ids[matched + edge_match_len]):
            edge_match_len += 1
        
        if edge_match_len < len(edge_tokens):
            # Partial match within this edge — must split
            child = self._split_node(child, edge_match_len)
            matched += edge_match_len
            node = child
            break  # can't go further
        
        # Full edge matched — continue to next level
        matched += len(edge_tokens)
        node = child
    
    return node, matched
```

**The split is the subtle part.** Think about what happens here:

```
Before split:
  root → [10, 20, 30, 40] (node A, has KV data for positions 0-3)

Query: [10, 20, 50, 60]
Match stops at position 2 (token 30 ≠ 50)

After split:
  root → [10, 20] (new node B, KV data for positions 0-1)
              → [30, 40] (old node A, KV data for positions 2-3)
```

Node B is created with the first `edge_match_len` tokens. Node A is shortened to the remaining tokens. B becomes A's parent. B gets the first half of A's KV data, A keeps the second half.

---

## Hint 3: Node Splitting — The Trickiest Part

This is where most bugs will live. You need to:

1. Create a new node (`mid_node`) for the matched prefix of the edge
2. Shorten the existing child to only contain the unmatched suffix
3. Re-parent: `mid_node` becomes the child of the original parent, and the shortened old child becomes a child of `mid_node`
4. Split the KV data: `mid_node` gets `kv[:, :split_len, :]`, old child gets `kv[:, split_len:, :]`

```python
def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
    """Split child's edge at position split_len. Returns the new mid-node."""
    mid_node = RadixNode()
    mid_node.token_ids = child.token_ids[:split_len]
    mid_node.parent = child.parent
    mid_node.last_access_time = child.last_access_time
    mid_node.lock_ref = child.lock_ref  # inherit lock from child
    
    # Split KV data
    if child.kv_data is not None:
        mid_node.kv_data = {}
        new_child_kv = {}
        for (layer, head), (k, v) in child.kv_data.items():
            mid_node.kv_data[(layer, head)] = (
                k[:, :split_len, :].clone(),
                v[:, :split_len, :].clone(),
            )
            new_child_kv[(layer, head)] = (
                k[:, split_len:, :].clone(),
                v[:, split_len:, :].clone(),
            )
        child.kv_data = new_child_kv
    
    # Re-wire the tree
    suffix_tokens = child.token_ids[split_len:]
    child.token_ids = suffix_tokens
    child.parent = mid_node
    
    # mid_node's child dict: keyed by first token of the suffix
    mid_node.children[suffix_tokens[0]] = child
    
    # Replace child in parent's children dict
    mid_node.parent.children[mid_node.token_ids[0]] = mid_node
    
    return mid_node
```

**Gotcha: `.clone()` the KV tensors.** If you just slice without cloning, both the mid_node and old child will hold views into the same underlying tensor. Later modifications (e.g., eviction freeing one node's data) would corrupt the other.

**Gotcha: lock_ref inheritance.** If the old child was pinned by an active request (`lock_ref > 0`), the new mid_node must also be pinned — otherwise the eviction logic might free the mid_node while the child's path is still in use.

---

## Hint 4: Insertion — Extending the Tree

After prefill, you commit new KV data into the tree. The insertion logic is similar to `match_prefix`, but when you reach the end of the matched path, you **create a new child** for the remaining tokens.

```python
def insert(self, token_ids: List[int], kv_data_full: Dict, block_size: int):
    """Insert token_ids and their KV data into the tree."""
    node, matched = self.match_prefix(token_ids)
    
    if matched == len(token_ids):
        return  # already fully in the tree
    
    # Create a new node for the unmatched suffix
    remaining = token_ids[matched:]
    new_node = RadixNode()
    new_node.token_ids = tuple(remaining)
    new_node.parent = node
    new_node.last_access_time = self.current_step
    
    # Extract the KV data for the new portion
    new_node.kv_data = {}
    for (layer, head), (k, v) in kv_data_full.items():
        new_node.kv_data[(layer, head)] = (
            k[:, matched:matched + len(remaining), :].clone(),
            v[:, matched:matched + len(remaining), :].clone(),
        )
    
    node.children[remaining[0]] = new_node
```

**Design decision: do you insert at block granularity or token granularity?**

In SGLang production, insertion is page-aligned (e.g., multiples of 16 tokens). For nanoGPT, you can keep it simple and insert at block_size granularity to match your existing system. But the tree itself doesn't care — it handles arbitrary-length edges.

---

## Hint 5: Loading KV Data from the Tree

Replace `load_cached_blocks()`. Instead of walking a hash chain, walk the tree from root to the matched node, collecting KV data along the path.

```python
def load_from_radix_tree(request, tree, prompt_tokens, block_size):
    """Load cached KV from the radix tree onto a request."""
    node, num_matched = tree.match_prefix(prompt_tokens)
    
    if num_matched == 0:
        return 0
    
    # Collect KV data by walking from root to matched node
    path_nodes = []
    current = node
    while current != tree.root:
        path_nodes.append(current)
        current = current.parent
    path_nodes.reverse()  # root-to-leaf order
    
    # Concatenate KV data along the path
    for (layer, head) in path_nodes[0].kv_data.keys():
        k_parts = []
        v_parts = []
        for pnode in path_nodes:
            pk, pv = pnode.kv_data[(layer, head)]
            k_parts.append(pk.clone())
            v_parts.append(pv.clone())
        request.kv_cache[(layer, head)] = (
            torch.cat(k_parts, dim=1),
            torch.cat(v_parts, dim=1),
        )
    
    # Pin all nodes on the path (inc lock_ref)
    for pnode in path_nodes:
        pnode.lock_ref += 1
    
    # Snap to block boundary for prefill_cursor
    num_cached = (num_matched // block_size) * block_size
    request.prefill_cursor = num_cached
    request._radix_path = path_nodes  # save for later unlock
    return num_cached
```

**Key insight: reference counting.** When you load KV from the tree, you `lock_ref += 1` on every node along the path. This tells the eviction logic "don't touch these nodes — an active request depends on them." When the request finishes, you walk the path and `lock_ref -= 1`.

**This is what SGLang calls `inc_lock_ref` / `dec_lock_ref`.** Look at the SGLang source — it walks from the leaf up to the root, incrementing/decrementing lock_ref on each node. Your version walks root-to-leaf, which is equivalent.

---

## Hint 6: Eviction — Leaves First, Never Orphan

This is the most important structural advantage over flat caching. The eviction rule:

1. **Only evict leaf nodes** (nodes with no children)
2. **Only evict unlocked nodes** (`lock_ref == 0`)
3. **Among eligible leaves, pick the one with the oldest `last_access_time`** (LRU)
4. **After evicting a leaf, check if its parent became a leaf** — if so, the parent is now a candidate for future eviction

```python
def evict_lru(self):
    """Evict the least-recently-used unlocked leaf node."""
    # Collect all evictable leaves
    leaves = []
    self._find_leaves(self.root, leaves)
    
    # Filter to unlocked leaves
    candidates = [n for n in leaves if n.lock_ref == 0 and n != self.root]
    if not candidates:
        return False  # nothing to evict
    
    victim = min(candidates, key=lambda n: n.last_access_time)
    
    # Remove from parent
    parent = victim.parent
    del parent.children[victim.token_ids[0]]
    
    # Free KV data
    victim.kv_data = None
    victim.parent = None
    
    return True

def _find_leaves(self, node, result):
    if not node.children:
        result.append(node)
    for child in node.children.values():
        self._find_leaves(child, result)
```

**Why this is better than flat LRU:** In the flat cache, evicting Block 5 silently breaks Blocks 6, 7, 8 (their chained hashes now lead nowhere). With the tree, Block 5 is an interior node with children — it can't be evicted until Blocks 6, 7, 8 are evicted first. The structural invariant is automatic.

**Performance note for nanoGPT:** Walking the whole tree to find leaves is fine at this scale. SGLang maintains a `evictable_leaves` set that's updated incrementally — an O(1) optimization you don't need here but should understand.

---

## Hint 7: Integrating with the Scheduler

The scheduler changes are minimal — you're swapping out the `BlockCache` for a `RadixTree`, but the interface is similar:

```python
class Scheduler:
    def __init__(self, ..., block_size=4):
        # CHANGED: radix tree instead of flat block cache
        self.radix_tree = RadixTree()
        self.block_size = block_size
    
    def _maybe_admit(self, step):
        # ... existing checks ...
        candidate = self.waiting[0]
        
        # CHANGED: use tree match instead of hash-chain walk
        _, num_cached = self.radix_tree.match_prefix(candidate.prompt_tokens)
        num_cached = (num_cached // self.block_size) * self.block_size
        
        actual_kv_cost = len(candidate.prompt_tokens) - num_cached
        if kv_used + actual_kv_cost > self.max_kv_tokens:
            return
        
        heapq.heappop(self.waiting)
        # CHANGED: load from tree
        load_from_radix_tree(candidate, self.radix_tree, candidate.prompt_tokens, self.block_size)
        candidate.status = "prefilling"
        self.prefilling.append(candidate)
```

In the generate loop, after prefill completes:
```python
if prefill_req.is_fully_prefilled:
    # CHANGED: insert into radix tree instead of committing blocks
    insert_into_radix_tree(prefill_req, scheduler.radix_tree, scheduler.block_size)
    # Unlock the path (the request is done with its cached prefix)
    unlock_radix_path(prefill_req)
    scheduler.promote(prefill_req)
```

And when a request finishes generation:
```python
for req in list(scheduler.active):
    if req.is_done:
        unlock_radix_path(req)  # release tree locks
        scheduler.complete(req)
```

---

## Hint 8: Visualizing the Tree (Debug Printing)

Add a `pretty_print()` method to see the tree structure. This is invaluable for debugging:

```python
def pretty_print(self, node=None, indent=0):
    if node is None:
        node = self.root
        print("RadixTree:")
    
    prefix = "  " * indent
    token_str = str(list(node.token_ids)[:8])
    if len(node.token_ids) > 8:
        token_str += "..."
    kv_str = f"KV[{node.kv_data is not None}]" if node != self.root else "ROOT"
    lock_str = f"lock={node.lock_ref}"
    print(f"{prefix}{token_str} ({kv_str}, {lock_str}, t={node.last_access_time})")
    
    for child in node.children.values():
        self.pretty_print(child, indent + 1)
```

Expected output after three requests with shared prefix `[1,2,3,4]`:

```
RadixTree:
[] (ROOT, lock=0, t=0)
  [1, 2, 3, 4] (KV[True], lock=0, t=5)
    [5, 6, 7, 8] (KV[True], lock=0, t=3)    ← req 0's suffix
    [9, 10, 11, 12] (KV[True], lock=1, t=5)  ← req 1's suffix (active)
    [13, 14] (KV[True], lock=0, t=4)          ← req 2's suffix
```

The shared prefix `[1,2,3,4]` appears **once** in the tree. Three different suffixes branch from it.

---

## Test Scenarios

### Test 1: Basic tree construction and matching
Insert `[1,2,3,4,5,6,7,8]`, then match against `[1,2,3,4,9,10]`. Should match 4 tokens (the shared prefix). The tree should split if needed.

### Test 2: Multi-branch sharing
Insert `[A, B, C, D, E, F]` and `[A, B, C, D, G, H]`. Both share `[A,B,C,D]`. The tree should have one path to `[A,B,C,D]` with two child branches. Verify KV data is shared, not duplicated.

### Test 3: Leaf-first eviction
Create a tree with a shared prefix and two branches. Evict one branch (leaf). Verify the other branch and shared prefix are untouched. Then evict the other branch. Now the shared prefix node should become a leaf and be evictable.

### Test 4: Lock reference counting
Start a request that loads KV from the tree (locks the path). Try to evict — the locked nodes should be protected. Finish the request (unlock). Now eviction should succeed.

### Test 5: Output correctness
Same as prefix caching — verify that generated text is identical with and without the radix tree. The tree is a cache optimization; it must not change model output.

### Test 6: Incremental insertion with branching workload
Simulate a multi-turn chat:
```
Turn 1: [sys, u1]         → insert path
Turn 2: [sys, u1, r1, u2] → extends existing path
Turn 3a: [sys, u1, r1, u3] → branches from Turn 2's prefix
Turn 3b: [sys, u1, r1, u4] → another branch
```
Verify the tree has the expected shape: one shared trunk with branches at the right points.

---

## Summary of Changes from Prefix Caching

| Component | What Changes |
|-----------|-------------|
| **Delete:** `hash_block_tokens()`, `NONE_HASH` | No more explicit hashing — tree structure encodes prefix chains |
| **Delete:** `CachedBlock`, `BlockCache` | Replaced by `RadixNode`, `RadixTree` |
| **Delete:** `find_cached_prefix()` | Replaced by `RadixTree.match_prefix()` |
| **New:** `RadixNode` class | Tree node with children, parent, token_ids, kv_data, lock_ref |
| **New:** `RadixTree` class | Tree with `match_prefix()`, `insert()`, `_split_node()`, `evict_lru()`, `pretty_print()` |
| **Rewrite:** `load_cached_blocks()` → `load_from_radix_tree()` | Walk tree path, collect KV, pin nodes with lock_ref |
| **Rewrite:** `commit_completed_blocks()` → `insert_into_radix_tree()` | Insert token+KV sequence into tree |
| **New:** `unlock_radix_path()` | Decrement lock_ref on request completion |
| **Modify:** `Scheduler.__init__()` | Replace `self.block_cache` with `self.radix_tree` |
| **Modify:** `Scheduler._maybe_admit()` | Use `radix_tree.match_prefix()` instead of `find_cached_prefix()` |
| **Modify:** generate loop | Use tree-based load/insert/unlock instead of block-based |
| **Modify:** `Request` dataclass | Add `_radix_path` field, remove `_committed_blocks` |
| Model / Head / assemble / disassemble | **Nothing changes** |

---

## Gotchas

1. **Clone KV on split.** When splitting a node, both halves must `.clone()` their KV tensors. Otherwise they share the same underlying storage, and freeing one corrupts the other.

2. **Lock_ref propagation on split.** If you split a locked node, the new mid-node must inherit the lock_ref count. Otherwise active requests might have their prefix evicted mid-generation.

3. **Children dict key must be an int (first token), not the full tuple.** Using the full tuple as key defeats the purpose — you'd need to scan all children to find a match.

4. **Don't forget to unlock.** Every `load_from_radix_tree` that pins nodes must have a corresponding `unlock_radix_path` when the request finishes. Forgetting this leaks lock_refs and prevents eviction — the tree fills up and nothing can be evicted.

5. **Edge case: empty tree.** On the first request, `match_prefix` returns `(root, 0)`. Make sure your insertion logic handles this cleanly — create a child of root with the full sequence.

6. **Edge case: exact match.** If the query exactly matches an existing path (no split needed), `match_prefix` returns the leaf node. Don't re-insert — just touch the `last_access_time`.

7. **Position alignment still matters.** Same as flat caching: KV tensors are position-dependent. A node storing KV for positions [4:8] can only be reused at positions [4:8]. The tree preserves this naturally because prefix paths always start at position 0.

---

## Implementation Checklist & Order

**Step 1: RadixNode and RadixTree skeleton**
- Implement `RadixNode` with children, parent, token_ids, kv_data, lock_ref, last_access_time.
- Implement `RadixTree` with root node and `pretty_print()`.

**Step 2: match_prefix**
- Implement tree traversal: walk edges, consuming query tokens.
- Handle full-edge matches (continue to next level) and end-of-query (return current node).
- **Do NOT implement splitting yet** — get the basic traversal working first.

**Step 3: _split_node**
- Implement the node split logic.
- Test it in isolation: insert `[1,2,3,4]`, then match `[1,2]` — verify the tree splits into `[1,2]` → `[3,4]`.
- Verify KV data is correctly divided and cloned.

**Step 4: insert**
- Call `match_prefix()` to find the divergence point.
- Create a new child node for the remaining tokens.
- Test: insert `[1,2,3,4]` then `[1,2,5,6]` — verify the tree has `[1,2]` → `{[3,4], [5,6]}`.

**Step 5: evict_lru**
- Find unlocked leaf nodes.
- Remove the LRU leaf and clean up the parent's children dict.
- Test: create a tree, evict, verify structure is correct.

**Step 6: load_from_radix_tree and lock_ref**
- Walk the matched path root-to-leaf, collect KV data, increment lock_ref.
- Implement `unlock_radix_path()` to decrement lock_ref.

**Step 7: Wire into the Scheduler**
- Replace `self.block_cache` with `self.radix_tree` in Scheduler.
- Update `_maybe_admit` to use `radix_tree.match_prefix()`.
- Update the generate loop to use tree-based load/insert/unlock.

**Step 8: Run Tests**
- Run all 6 test scenarios above.
- Run the existing benchmark suite (adapted for radix tree).
- Verify output correctness against non-caching baseline.
