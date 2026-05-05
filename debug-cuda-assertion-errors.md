# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

============================================================
Test 1: Single request with Request object
============================================================
---------------------------------------------------------------------------
AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_3637/1909429672.py in <cell line: 0>()
      9     max_new_tokens=200,
     10 )
---> 11 generate_request(model, req)
     12 
     13 print(f"Status: {req.status}")

/tmp/ipykernel_3637/1890930242.py in generate_request(model, request)
     68             idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
     69 
---> 70             request.generated_tokens.append(idx_next.item())
     71 
     72             if request.is_done:

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

I got this error when running the # ── Multiple independent requests with different prompts/lengths ─────────────

how do i fix this @[nanogpt-cont-batching.ipynb]

*User accepted the command `jupyter nbconvert --to script nanogpt-cont-batching.ipynb --stdout 2>/dev/null`*

*User accepted the command `python3 -c "import json; nb = json.load(open('nanogpt-cont-batching.ipynb')); [print(f'--- Cell {i} ---\n' + ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type']=='code']"`*

### Planner Response

Now let me see the earlier cells too, especially the `generate_request` function and the `Request` class:

### User Input

============================================================
Test 1: Single request with Request object
============================================================
---------------------------------------------------------------------------
AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_3637/1909429672.py in <cell line: 0>()
      9     max_new_tokens=200,
     10 )
---> 11 generate_request(model, req)
     12 
     13 print(f"Status: {req.status}")

/tmp/ipykernel_3637/1890930242.py in generate_request(model, request)
     68             idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
     69 
---> 70             request.generated_tokens.append(idx_next.item())
     71 
     72             if request.is_done:

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

I got this error when running the # ── single request test cell

how do i fix this @[nanogpt-cont-batching.ipynb]

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt-cont-batching.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        print(f'--- Cell {i} ---')
        print(src)
        print()
"`*

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt-cont-batching.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and i < 17:
        src = ''.join(c['source'])
        print(f'--- Cell {i} ---')
        print(src)
        print()
"`*

### Planner Response

I can see the bug. Look at this line in the `generate_request` decode loop:

```python
curr_pos = torch.tensor(
    [[len(request.tokens_so_far) - 1]], device=device
)
```

The `tokens_so_far` property includes only the **prompt + already-generated tokens**, but it appends `idx_next.item()` **before** this position is computed. Let me check the `Request` class and `tokens_so_far`:

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt-cont-batching.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and i < 10:
        src = ''.join(c['source'])
        print(f'--- Cell {i} ---')
        print(src)
        print()
"`*

### Planner Response

Let me find the `Request` class definition:

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt-cont-batching.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'class Request' in src or 'tokens_so_far' in src:
        print(f'--- Cell {i} ({c[\"cell_type\"]}) ---')
        print(src)
        print()
"`*

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt-cont-batching.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'class Request' in src:
        print(f'--- Cell {i} ---')
        print(src)
"`*

### Planner Response

Now I have a clear picture. Let me trace through the bug precisely.

---

The root cause is an **off-by-one in `curr_pos`** inside `generate_request`. Here's the execution order in the decode loop:

```python
# Step N of decode loop:
request.generated_tokens.append(idx_next.item())   # ← token appended FIRST

# tokens_so_far is now prompt + N generated tokens
# len(tokens_so_far) - 1 = prompt_len + N - 1

curr_pos = torch.tensor([[len(request.tokens_so_far) - 1]], device=device)
```

After appending, `tokens_so_far` has length `prompt_len + N`. So `curr_pos = prompt_len + N - 1`.

But what position is `idx_next` actually at? It's the token being **fed into the next decode step**, which sits at position `prompt_len + N`. So `curr_pos` is **one too small**.

As `N` grows toward `max_new_tokens=200`, the over-large `block_size` is fine, but the position embedding table has only `block_size` entries. Once `curr_pos` becomes ≥ `block_size`, `nn.Embedding` triggers an out-of-bounds index → **CUDA device-side assert**.

**The fix:** compute `curr_pos` *before* appending the token, or add 1 to account for the already-appended token:

```python
# Option A: compute position before appending (cleanest)
curr_pos = torch.tensor([[len(request.tokens_so_far)]], device=device)  # prompt_len + N = correct next position
request.generated_tokens.append(idx_next.item())
```

Or equivalently:

```python
# Option B: keep append first, then correct the +1 offset
request.generated_tokens.append(idx_next.item())
curr_pos = torch.tensor([[len(request.tokens_so_far)]], device=device)  # no more - 1
```

In your notebook cell 10, `generate_request`'s decode loop looks like this — here's the fix (Option B, minimal change):

**Find this block** (around lines 65–75 of cell 10):
```python
request.generated_tokens.append(idx_next.item())

if request.is_done:
    break

# Rebuild past_kvs ...
curr_pos = torch.tensor(
    [[len(request.tokens_so_far) - 1]], device=device
)  # (1, 1)
```

**Change to:**
```python
request.generated_tokens.append(idx_next.item())

if request.is_done:
    break

# Rebuild past_kvs ...
curr_pos = torch.tensor(
    [[len(request.tokens_so_far)]], device=device   # ← remove the - 1
)  # (1, 1)
```

**Why this works:** After appending, `tokens_so_far` has length `prompt_len + N`. The *next* token to generate goes at position `prompt_len + N`, which is exactly `len(tokens_so_far)`. The old `- 1` made you look up the position of the token you *just generated*, not the one you're about to generate — causing the position index to lag by 1 every step, and eventually walk off the end of the position embedding table when prompts are long or generation is long.

### User Input

============================================================
Test 1: Single request with Request object
============================================================
---------------------------------------------------------------------------
AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_3637/1909429672.py in <cell line: 0>()
      9     max_new_tokens=200,
     10 )
---> 11 generate_request(model, req)
     12 
     13 print(f"Status: {req.status}")

/tmp/ipykernel_3637/1932475304.py in generate_request(model, request)
     47     with torch.no_grad():
     48         # Convert prompt to tensor
---> 49         prompt = torch.tensor(
     50             [request.prompt_tokens], dtype=torch.long, device=device
     51         )  # (1, T_prompt)

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

i'm getting this error now after removing the 1

### Planner Response

This is a **CUDA sticky error** — not a new bug in your code. The error message even hints at it:

> *"CUDA kernel errors might be asynchronously reported at some other API call"*

Once a device-side assert fires (like the out-of-bounds embedding index from before), the CUDA context for that entire process becomes permanently poisoned. Every subsequent GPU call — even trivial ones like `torch.tensor(...)` — will immediately fail with the same `AcceleratorError`. The fix to `curr_pos` is correct, but the kernel is still in the broken state from the previous crash.

**The fix is simply to restart the kernel and re-run all cells from scratch.**

In Jupyter:
- **Kernel → Restart & Run All** (or `Kernel → Restart Kernel…` then re-run manually)

After restart, the CUDA context will be fresh, your `curr_pos` fix will be in effect from the start, and the error won't recur.

### User Input

============================================================
Test 1: Single request with Request object
============================================================
---------------------------------------------------------------------------
AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_6876/1909429672.py in <cell line: 0>()
      9     max_new_tokens=200,
     10 )
---> 11 generate_request(model, req)
     12 
     13 print(f"Status: {req.status}")

/tmp/ipykernel_6876/1932475304.py in generate_request(model, request)
     68             idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
     69 
---> 70             request.generated_tokens.append(idx_next.item())
     71 
     72             if request.is_done:

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

i'm still getting this error on that exact same cell

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt-cont-batching.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'block_size' in src and i < 5:
        print(f'--- Cell {i} ---')
        print(src)
        print()
"`*

### Planner Response

There it is — **`block_size = 32`**.

My original analysis was wrong about the `- 1`. Let me retrace:

- The `- 1` **is correct** — `len(tokens_so_far) - 1` gives the 0-indexed position of the last token, which matches `generate_with_cache`.
- **Revert the `- 1` removal** — the original code was right there.

The **real bug** is that `block_size = 32` but you're generating `max_new_tokens=200`. The position embedding table only has 32 entries (`nn.Embedding(block_size, n_embd)`), so once the sequence exceeds 32 tokens, `curr_pos ≥ 32` causes an out-of-bounds lookup in the position embedding table → CUDA assert.

**The fix:** reduce `max_new_tokens` so that `len(prompt) + max_new_tokens ≤ block_size (32)`:

In the test cell (Cell 12), change:
```python
# Was:
req = Request(
    id=0,
    prompt_tokens=[0],
    max_new_tokens=200,  # ← way too big for block_size=32
)

# Fix:
req = Request(
    id=0,
    prompt_tokens=[0],
    max_new_tokens=31,   # 1 prompt + 31 generated = 32 = block_size
)
```

And **undo the `- 1` removal** — put it back to:
```python
curr_pos = torch.tensor(
    [[len(request.tokens_so_far) - 1]], device=device
)
```

The `generate_with_cache` function doesn't have this bug because it truncates the prompt with `idx[:, -block_size:]`, but `generate_request` doesn't have that guard. You should also add the same guard to `generate_request` if you want to allow longer generation in the future, but for now just keeping `max_new_tokens` within `block_size` is the fix.

**TL;DR:** The position embedding table has only 32 slots. Generating 200 tokens tries to look up position 32+, which crashes CUDA. Set `max_new_tokens` ≤ `block_size - len(prompt)`.

### User Input

ok, but i don't want to restart my kernel since i am training a model in previous cells. how do i get around this? 

### Planner Response

Once a CUDA device-side assert fires, the GPU context is permanently poisoned for that process — there's no way to clear it. But you don't have to lose your trained model. Just **save it to disk**, restart, and reload:

**Run this cell now (it will work even with a poisoned CUDA context since it just moves to CPU and saves):**

```python
# Save model to disk before restarting
torch.save(model.cpu().state_dict(), "model_checkpoint.pt")
print("✓ Model saved to model_checkpoint.pt")
```

Then restart the kernel, re-run the setup/definition cells, and **replace** the training cell with:

```python
model = GPTLanguageModel()
model.load_state_dict(torch.load("model_checkpoint.pt"))
m = model.to(device)
print("✓ Model loaded from checkpoint")
```

That way you skip the training but get the exact same model weights back in a clean CUDA context, and your fixed cells will work.

### User Input

============================================================
Test 2: Multiple independent requests
============================================================
---------------------------------------------------------------------------
AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_9671/2977180656.py in <cell line: 0>()
     11 
     12 for req in requests:
---> 13     generate_request(model, req)
     14     print(f"\n--- Request {req.id} ({req.status}, {req.num_generated} tokens) ---")
     15     print(decode(req.tokens_so_far))

/tmp/ipykernel_9671/1890930242.py in generate_request(model, request)
     68             idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
     69 
---> 70             request.generated_tokens.append(idx_next.item())
     71 
     72             if request.is_done:

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

how do i fix this

### Planner Response

Same root cause — `block_size = 32` but your `max_new_tokens` values exceed it. Let me check the math for Cell 13:

| Request | Prompt | Prompt tokens | max_new_tokens | Total | vs block_size=32 |
|---------|--------|--------------|----------------|-------|------------------|
| 0 | `"O Romeo, "` | 9 | **50** | 59 | ❌ overflow |
| 1 | `"To be or "` | 9 | **100** | 109 | ❌ overflow |
| 2 | `"KING HENRY:"` | 11 | **30** | 41 | ❌ overflow |

Every request blows past the 32-position embedding table. Fix by capping `max_new_tokens` so `prompt_len + max_new_tokens ≤ 32`:

```python
requests = [
    Request(id=0, prompt_tokens=encode("O Romeo, "),     max_new_tokens=23),  # 9 + 23 = 32
    Request(id=1, prompt_tokens=encode("To be or "),     max_new_tokens=23),  # 9 + 23 = 32
    Request(id=2, prompt_tokens=encode("KING HENRY:"),   max_new_tokens=21),  # 11 + 21 = 32
]
```

And again — since the previous crash poisoned the CUDA context (new kernel PID 9671, but the crash happened within this same session), you'll need to **save → restart → reload** before this will run:

```python
torch.save(model.cpu().state_dict(), "model_checkpoint.pt")
```

Then restart and reload. You'll need to do this every time a CUDA assert fires — there's no recovery without restarting the Python process.

**The same fix applies to all your other test cells** — Cell 12 (already fixed to 31), Cell 15 (benchmark with `N_TOKENS=200`), and Cell 18 (continuous batching). Every `max_new_tokens` value must satisfy `prompt_len + max_new_tokens ≤ 32`.

### User Input

@[nanogpt_cont_batching.ipynb]could you look at the entire notebook and tell me if this problem still exists for any cell?

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt_cont_batching.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        print(f'--- Cell {i} ---')
        print(src)
        print()
"`*

### Planner Response

Now let me see the first half of the cells (the output was truncated):

*User accepted the command `python3 -c "
import json
nb = json.load(open('nanogpt_cont_batching.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and i >= 9 and i <= 16:
        src = ''.join(c['source'])
        print(f'--- Cell {i} ---')
        print(src)
        print()
"`*

### Planner Response

Here's the full audit. With `block_size = 32`, every cell needs `prompt_len + max_new_tokens ≤ 32`:

| Cell | What | Prompt len | max_new_tokens | Total | Status |
|------|------|-----------|----------------|-------|--------|
| **12** | Single request test | 1 | 31 | 32 | ✅ Fixed |
| **13** | Multiple requests | 9, 9, 11 | 23, 23, 21 | 32, 32, 32 | ✅ Fixed |
| **15** | Benchmark | 1 | 10 | 11 | ✅ OK |
| **15** | Benchmark warmup | 1 | 10 | 11 | ✅ OK |
| **18** | Continuous batching | 9, 9, 12 | 22, 22, 15 | 31, 31, 27 | ✅ Fixed |

Also checked `generate_with_cache` (Cell 10) — it has the same overflow risk but `N_TOKENS=10` in cell 15 keeps it safe.

**One remaining bug** in `generate_with_cache` (Cell 10): unlike `generate_no_cache` which truncates with `idx[:, -block_size:]`, `generate_with_cache` passes the full prompt without truncation. If someone ever calls it with a long prompt, the prefill would overflow. But with the current test values it's fine.

**All cells look safe now.** After your checkpoint-save/kernel-restart/reload, they should all run cleanly.

### User Input

============================================================
Test 2: Multiple independent requests
============================================================

--- Request 0 (done, 23 tokens) ---
O Romeo, he couriny: thy have th

--- Request 1 (done, 23 tokens) ---
To be or lety.
Herefore you pati

--- Request 2 (done, 21 tokens) ---
KING HENRY:
Ay, who't you, thoug
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_15967/1036277605.py in <cell line: 0>()
     24     sample_k, _ = req.kv_cache[(0, 0)]  # layer 0, head 0
     25     expected_T = len(req.prompt_tokens) + req.num_generated
---> 26     assert sample_k.shape[1] == expected_T, (
     27         f"Request {req.id}: cache T={sample_k.shape[1]}, expected {expected_T}"
     28     )

AssertionError: Request 0: cache T=31, expected 32

### Planner Response

The generation succeeded — this is just an assertion bug. Here's why:

In `generate_request`, the last decode iteration does:
1. Sample `idx_next` from logits
2. Append to `generated_tokens` → count = 23
3. `is_done` → True → **break** (never feeds the last token through the model)

So the cache has entries for every token that was *processed by the model*, which is `prompt + (num_generated - 1)` — the final sampled token never gets a forward pass because generation is done.

```
Prefill:  9 tokens processed  → cache T = 9
Decode 1: feed token, sample  → cache T = 10
...
Decode 22: feed token, sample → cache T = 31
Decode 23: sample, is_done, break → cache stays T = 31
```

**Fix the assertion** in Cell 13 (and Cell 12 if present, and Cell 18):

```python
# Was:
expected_T = len(req.prompt_tokens) + req.num_generated

# Fix:
expected_T = len(req.prompt_tokens) + req.num_generated - 1
```

The cache is correct — the last generated token was sampled but never needed to be fed back into the model, so it has no KV entry. That's expected behavior.

### User Input

============================================================
Continuous Batching — Simulated Arrivals
============================================================
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_15967/3800753328.py in <cell line: 0>()
     10 print("=" * 60)
     11 
---> 12 completed = continuous_batching_generate(model, request_queue, max_batch_size=4)
     13 
     14 # Print results

7 frames
/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
   2565         # remove once script supports set_grad_enabled
   2566         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
-> 2567     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
   2568 
   2569 

RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)

### Planner Response

Two tensors in `continuous_batching_generate` are missing `device=device`:

**1.** Prefill prompt tensor:
```python
# Bug:
prompt = torch.tensor([req.prompt_tokens])

# Fix:
prompt = torch.tensor([req.prompt_tokens], device=device)
```

**2.** Batch positions tensor:
```python
# Bug:
batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in active_requests])

# Fix:
batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in active_requests], device=device)
```

Both default to CPU, but the model weights and embeddings are on CUDA. Add `device=device` to both and it should work.

### User Input

============================================================
Continuous Batching — Simulated Arrivals
============================================================
  [step 0] Admitted request 0 (prompt=9, max_new=22)
  [step 0] Admitted request 1 (prompt=9, max_new=22)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_15967/3800753328.py in <cell line: 0>()
     10 print("=" * 60)
     11 
---> 12 completed = continuous_batching_generate(model, request_queue, max_batch_size=4)
     13 
     14 # Print results

/tmp/ipykernel_15967/536747971.py in continuous_batching_generate(model, request_queue, max_batch_size)
    130             B_active = len(active_requests)
    131 
--> 132             batch_tokens = torch.cat([req._last_token for req in active_requests], device=device)
    133 
    134             batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in active_requests], device=device)

TypeError: cat() received an invalid combination of arguments - got (list, device=str), but expected one of:
 * (tuple of Tensors tensors, int dim = 0, *, Tensor out = None)
 * (tuple of Tensors tensors, name dim, *, Tensor out = None)


### Planner Response

`torch.cat` doesn't take a `device` argument — it just concatenates tensors that are already on the right device. The `_last_token` tensors are already on CUDA (they came from `torch.multinomial`), so just remove it:

```python
# Fix:
batch_tokens = torch.cat([req._last_token for req in active_requests])
```

That line was correct originally — only `torch.tensor(...)` needed the `device=device` addition.

### User Input

============================================================
Continuous Batching — Simulated Arrivals
============================================================
  [step 0] Admitted request 0 (prompt=9, max_new=22)
  [step 0] Admitted request 1 (prompt=9, max_new=22)
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_15967/3800753328.py in <cell line: 0>()
     10 print("=" * 60)
     11 
---> 12 completed = continuous_batching_generate(model, request_queue, max_batch_size=4)
     13 
     14 # Print results

1 frames
/tmp/ipykernel_15967/91558622.py in assemble_batch_cache(requests)
     14 
     15     B = len(requests)
---> 16     lengths = [req.kv_cache[(0, 0)[0]].shape[1] for req in requests]
     17     max_t = max(lengths)
     18 

KeyError: 0

### Planner Response

Operator precedence bug. This line:

```python
lengths = [req.kv_cache[(0, 0)[0]].shape[1] for req in requests]
```

`(0, 0)[0]` evaluates to `0` (indexing into the tuple), so it's looking up `req.kv_cache[0]` instead of `req.kv_cache[(0, 0)]`. Fix:

```python
lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
```

This does: cache key `(0, 0)` → get the `(k, v)` tuple → `[0]` picks `k` → `.shape[1]` gets sequence length.

### User Input

============================================================
Continuous Batching — Simulated Arrivals
============================================================
  [step 0] Admitted request 0 (prompt=9, max_new=22)
  [step 0] Admitted request 1 (prompt=9, max_new=22)
  [step 3] Admitted request 2 (prompt=12, max_new=15)
  [step 16] Completed request 2 (15 tokens)
  [step 20] Completed request 0 (22 tokens)
  [step 20] Completed request 1 (22 tokens)

────────────────────────────────────────
Request 0  |  22 tokens  |  status: done
────────────────────────────────────────
O Romeo, your from careween par

────────────────────────────────────────
Request 1  |  22 tokens  |  status: done
────────────────────────────────────────
To be or had catple.
The own 'A

────────────────────────────────────────
Request 2  |  15 tokens  |  status: done
────────────────────────────────────────
KING HENRY:
I leaves, and h
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_15967/3800753328.py in <cell line: 0>()
     23     k, _ = req.kv_cache[(0, 0)]
     24     expected_T = len(req.prompt_tokens) + req.num_generated
---> 25     assert k.shape[1] == expected_T, f"Req {req.id}: cache T={k.shape[1]}, expected {expected_T}"
     26     assert req.status == "done"
     27     assert req.num_generated == req.max_new_tokens

AssertionError: Req 2: cache T=26, expected 27

### Planner Response

Same assertion fix as Cell 13 — the last sampled token is never fed through the model, so the cache is always 1 short:

```python
# Was:
expected_T = len(req.prompt_tokens) + req.num_generated

# Fix:
expected_T = len(req.prompt_tokens) + req.num_generated - 1
```

The continuous batching itself is working perfectly — all 3 requests completed correctly with staggered arrivals! 🎉