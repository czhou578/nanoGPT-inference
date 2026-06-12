# Disaggregated Prefill — Implementation Plan & Hints

## The Problem You're Solving

Your current inference engine runs prefill and decode **in the same thread, on the same model**. Every scheduler step does:

1. One chunked-prefill pass (compute-bound, processes many tokens)
2. One batched-decode pass (memory-bound, processes 1 token per request)

These two phases have completely different hardware profiles:
- **Prefill** is compute-bound — high arithmetic intensity, saturates FLOPs
- **Decode** is memory-bandwidth-bound — fetches the entire KV cache to produce 1 token

Running them on the same device means they **interfere** with each other. While prefill is crunching through tokens, decode requests are stalled. While decode is streaming 1-token-at-a-time, the GPU compute is underutilized.

**Disaggregated prefill** solves this by splitting them into separate workers. In production systems (vLLM's PD disaggregation, DeepSeek's approach), these are separate GPU instances. On a single machine, you can simulate the architectural pattern using **separate threads/processes with KV cache handoff via shared memory**.

---

## Base File

**Start from: [`nanogpt-radix-tree.py`](../../nanogpt-radix-tree.py)**

This is the right base because:
- It has the most complete `Request` dataclass (with `prefill_cursor`, `kv_cache`, `is_fully_prefilled`)
- It has a proper `Scheduler` with `waiting → prefilling → active → done` state machine
- It has `assemble_batch_cache()` / `disassemble_batch_cache()` for batched decode
- It has radix tree prefix caching (which naturally fits disaggregated prefill — the prefill worker populates the tree)
- It has `insert_into_radix_tree()` / `load_from_radix_tree()` for KV cache sharing

You'll copy this file as `nanogpt-disaggregated-prefill.py` and modify it.

---

## Hint 1: Understand the Architecture

In disaggregated prefill, you split the engine into two independent workers:

```
                     ┌─────────────────────────┐
   incoming ───────►│    PREFILL WORKER         │
   requests          │  (Thread 1)               │
                     │  • Runs full prefill pass  │
                     │  • Produces KV cache       │
                     │  • Writes to shared memory │
                     └──────────┬──────────────┘
                                │
                         KV cache handoff
                       (shared memory / queue)
                                │
                     ┌──────────▼──────────────┐
                     │    DECODE WORKER          │
                     │  (Thread 2)               │
                     │  • Reads KV from shared   │
                     │  • Runs autoregressive     │
                     │    decode loop             │
                     │  • Produces output tokens  │
                     └───────────────────────────┘
```

Each worker has its **own copy of the model** (or in production, its own GPU). The key challenge is the **KV cache handoff**: after prefill completes, the KV cache must be transferred to the decode worker efficiently.

On a single machine with `torch.multiprocessing`, you can use `torch.Tensor` in shared memory (via `tensor.share_memory_()`) or simply pass serialized tensors through a `multiprocessing.Queue`.

**Question to ask yourself:** Why can't you just use Python threads with the GIL? (Answer: PyTorch releases the GIL during tensor ops, so two threads running `model(...)` can genuinely overlap. But for cleaner separation, `threading.Thread` is sufficient for the educational demo since both threads release the GIL during the forward pass.)

---

## Hint 2: Design the KV Transfer Protocol

You need a data structure to hand off from prefill → decode. Define a `KVTransfer` dataclass:

```python
@dataclass
class KVTransfer:
    """Payload sent from prefill worker to decode worker."""
    request_id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
    first_token_id: int          # the token sampled at the end of prefill
    prefill_time_ms: float       # for latency tracking
```

The prefill worker puts a `KVTransfer` into a thread-safe queue. The decode worker reads from the queue and constructs a `Request` with the pre-filled KV cache already loaded.

**Key design decision:** The KV tensors here are regular `torch.Tensor` objects. In a real multi-GPU system, this would be a network transfer (NCCL, RDMA, etc.). Your implementation simulates the same semantic boundary — the decode worker should NOT assume it can access the prefill worker's state directly.

---

## Hint 3: Build the Prefill Worker

The prefill worker is a loop that:

1. Reads incoming requests from a `request_queue`
2. Runs the full forward pass (no chunking — the prefill worker can process the entire prompt at once since it doesn't share a budget with decode)
3. Samples the first token
4. Packs the KV cache into a `KVTransfer` object
5. Puts it on the `kv_transfer_queue` for the decode worker

```python
def prefill_worker(model, request_queue, kv_transfer_queue, stop_event):
    """
    Runs in a separate thread. Pulls requests from request_queue,
    runs full prefill, sends KV cache to decode worker.
    """
    model.eval()
    with torch.no_grad():
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            
            # Full prefill — no chunking needed since this worker
            # isn't sharing a token budget with decode
            prompt = torch.tensor([request.prompt_tokens], device=device)
            logits, _, new_kvs = model(prompt)

            # Build per-(layer, head) KV cache dict
            kv_cache = {}
            for li, bkv in enumerate(new_kvs):
                for hi, (k, v) in enumerate(bkv):
                    kv_cache[(li, hi)] = (k.clone(), v.clone())
            
            # Sample first token
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            first_token = torch.multinomial(probs, num_samples=1)

            prefill_ms = (time.perf_counter() - t0) * 1000

            transfer = KVTransfer(
                request_id=request.id,
                prompt_tokens=request.prompt_tokens,
                max_new_tokens=request.max_new_tokens,
                kv_cache=kv_cache,
                first_token_id=first_token.item(),
                prefill_time_ms=prefill_ms,
            )

            kv_transfer_queue.put(transfer)
```

Notice that the prefill worker runs the **entire prompt in one pass** — there's no chunking because it doesn't share a token budget with decode. This is one of the key benefits of disaggregation: the prefill worker can saturate its compute without throttling to leave room for decode tokens.

---

## Hint 4: Build the Decode Worker

The decode worker is a continuous batching loop (very similar to your existing `scheduled_generate`), but instead of doing its own prefill, it receives pre-filled KV caches from the transfer queue:

```python
def decode_worker(model, kv_transfer_queue, results_queue, stop_event,
                  max_batch_size=4):
    """
    Runs in a separate thread. Receives pre-filled requests from
    kv_transfer_queue, runs autoregressive decode, sends completed
    requests to results_queue.
    """
    model.eval()
    active_requests = []

    with torch.no_grad():
        while not stop_event.is_set() or active_requests:
            # Drain the transfer queue — admit new pre-filled requests
            while not kv_transfer_queue.empty() and len(active_requests) < max_batch_size:
                transfer = kv_transfer_queue.get()
                req = Request(
                    id=transfer.request_id,
                    prompt_tokens=transfer.prompt_tokens,
                    max_new_tokens=transfer.max_new_tokens,
                    kv_cache=transfer.kv_cache,
                    status="active",
                )
                req.generated_tokens.append(transfer.first_token_id)
                req._last_token = torch.tensor([[transfer.first_token_id]], device=device)
                req.prefill_cursor = len(req.prompt_tokens)  # fully prefilled
                active_requests.append(req)

            if not active_requests:
                time.sleep(0.01)
                continue

            # Standard batched decode — identical to your existing code
            batch_tokens = torch.cat([r._last_token for r in active_requests])
            batch_positions = torch.tensor(
                [[len(r.tokens_so_far) - 1] for r in active_requests],
                device=device,
            )

            past_kvs, attn_mask, pad_lengths = assemble_batch_cache(active_requests)
            logits, _, new_kvs = model(
                batch_tokens, pos=batch_positions,
                past_kvs=past_kvs, attn_mask=attn_mask,
            )

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            disassemble_batch_cache(active_requests, new_kvs, pad_lengths)

            for i, req in enumerate(active_requests):
                req.generated_tokens.append(idx_next[i].item())
                req._last_token = idx_next[i:i + 1]

            still_active = []
            for req in active_requests:
                if req.is_done:
                    results_queue.put(req)
                else:
                    still_active.append(req)
            active_requests = still_active
```

**Key insight:** The decode worker's loop is almost identical to `continuous_batching_generate()` — the only difference is that instead of doing prefill itself, it receives pre-filled KV caches from the transfer queue. The decode worker **never runs a prefill forward pass**.

---

## Hint 5: Wire It Together with a Coordinator

The main function creates both workers and feeds requests:

```python
def disaggregated_generate(model, requests, max_batch_size=4):
    """
    Disaggregated prefill/decode: two workers, KV cache handoff via queue.
    """
    import threading
    import queue

    request_queue = queue.Queue()       # main → prefill worker
    kv_transfer_queue = queue.Queue()   # prefill worker → decode worker
    results_queue = queue.Queue()       # decode worker → main
    stop_event = threading.Event()

    # Both workers use the SAME model (shared weights, different forward passes)
    prefill_thread = threading.Thread(
        target=prefill_worker,
        args=(model, request_queue, kv_transfer_queue, stop_event),
        daemon=True,
    )
    decode_thread = threading.Thread(
        target=decode_worker,
        args=(model, kv_transfer_queue, results_queue, stop_event, max_batch_size),
        daemon=True,
    )

    prefill_thread.start()
    decode_thread.start()

    # Feed requests
    for req in requests:
        request_queue.put(req)

    # Collect results
    completed = []
    while len(completed) < len(requests):
        result = results_queue.get(timeout=30)
        completed.append(result)

    stop_event.set()
    prefill_thread.join(timeout=5)
    decode_thread.join(timeout=5)

    return completed
```

**Important:** Both threads share the same `model` object. This works because:
1. PyTorch releases the GIL during tensor operations
2. The model weights are read-only during `eval()` mode
3. Each thread creates its own tensors (no shared mutable state except the queues)

In a real production system (vLLM P/D), each worker would have its **own GPU** with its own model copy. The shared-model approach here simulates the same architectural boundary without needing multiple GPUs.

---

## Hint 6: Add Telemetry to Make Disaggregation Visible

The whole point of this exercise is to **see** the disaggregation in action. Add per-step logging that shows the two workers operating independently:

```python
# In prefill_worker:
print(f"  [PREFILL] req {req.id}: {len(req.prompt_tokens)} tokens → "
      f"KV transfer ({prefill_ms:.1f}ms)")

# In decode_worker:
print(f"  [DECODE]  step {step}: batch={len(active_requests)} "
      f"({', '.join(f'req{r.id}' for r in active_requests)})")
```

You should see output like:
```
  [PREFILL] req 0: 30 tokens → KV transfer (2.3ms)
  [PREFILL] req 1: 15 tokens → KV transfer (1.1ms)
  [DECODE]  step 0: batch=1 (req0)
  [DECODE]  step 1: batch=2 (req0, req1)
  [PREFILL] req 2: 20 tokens → KV transfer (1.5ms)
  [DECODE]  step 2: batch=2 (req0, req1)
  [DECODE]  step 3: batch=3 (req0, req1, req2)
  ...
```

Notice how prefill and decode steps **interleave asynchronously** — prefill doesn't block decode, and decode doesn't wait for prefill.

---

## Hint 7: Add a Baseline Comparison

To demonstrate the benefit, compare disaggregated vs monolithic (your existing `scheduled_generate`):

```python
# Monolithic: prefill and decode share one thread
t0 = time.perf_counter()
mono_results = scheduled_generate(model, requests_mono, ...)
mono_time = time.perf_counter() - t0

# Disaggregated: separate prefill and decode workers
t0 = time.perf_counter()
disagg_results = disaggregated_generate(model, requests_disagg, ...)
disagg_time = time.perf_counter() - t0
```

The disaggregated version should show:
- **Lower TTFT** (time-to-first-token) for decode-phase requests — they aren't blocked by prefill
- **Overlap** — new requests start prefilling while existing requests are decoding
- **Same output correctness** — the tokens produced should be identical (given the same seed)

---

## Hint 8: Staggered Arrivals to Show the Benefit

The disaggregation advantage is most visible with **staggered arrivals** where some requests arrive while others are already decoding:

```python
requests = [
    Request(id=0, prompt_tokens=encode("First Citizen: " * 10), max_new_tokens=20),
    Request(id=1, prompt_tokens=encode("ROMEO: " * 5), max_new_tokens=15),
    Request(id=2, prompt_tokens=encode("JULIET: " * 8), max_new_tokens=10),
]
```

In the monolithic scheduler, req 1's decode is **blocked** while req 2 is being prefilled. In the disaggregated version, the decode worker keeps generating tokens for req 0 and req 1 **while the prefill worker is processing req 2's prompt**.

---

## Test Scenarios

### Test 1: Correctness — Output Equivalence
Run the same requests through both `scheduled_generate` (monolithic) and `disaggregated_generate`. With the same seed and sampling, the generated tokens should be **identical**. This proves that the KV cache handoff didn't corrupt anything.

### Test 2: Overlap Visibility
Submit 4+ requests and verify from logs that:
- Prefill worker processes requests concurrently with decode
- Decode batch grows as prefilled requests arrive
- No request's decode is stalled by another's prefill

### Test 3: Latency Comparison
Measure TTFT (time from request submission to first generated token) and total generation time. The disaggregated version should have lower TTFT for later-arriving requests because decode doesn't block behind prefill.

### Test 4: Stress Test
Submit 16+ requests and verify:
- All requests complete
- No deadlocks or queue starvation
- Memory usage is reasonable (KV tensors are correctly freed after completion)

---

## Summary of New Components vs Radix Tree

| Component | What's New |
|-----------|------------|
| `KVTransfer` dataclass | New — payload for prefill→decode handoff |
| `prefill_worker()` | New — runs full prefill in a dedicated thread |
| `decode_worker()` | New — runs batched decode in a dedicated thread |
| `disaggregated_generate()` | New — coordinator that wires the two workers together |
| `Request` dataclass | Unchanged |
| Model / Head / assemble_batch_cache | **Nothing changes** — disaggregation is pure Python above the model |
| `Scheduler` | Simplified — the decode worker doesn't need admission/preemption for prefill |

The key insight: **the model doesn't know anything about disaggregation**. All of this complexity lives in the threading and queue orchestration above it. The `model.forward()` call is identical whether it's running prefill or decode — it's just called from different threads with different inputs.

---

## Recommended Implementation Order

1. **Step 1: Copy `nanogpt-radix-tree.py` → `nanogpt-disaggregated-prefill.py`**
   - Update the module docstring to describe disaggregated prefill.
   - Remove the radix tree benchmark runner at the bottom.

2. **Step 2: Define `KVTransfer` dataclass (Hint 2)**
   - This is the contract between the two workers.
   - Keep it simple — just the request metadata + serialized KV cache.

3. **Step 3: Implement `prefill_worker()` (Hint 3)**
   - This is the simpler of the two workers — just a loop that reads from a queue, runs `model(prompt)`, and writes to the transfer queue.
   - Test it standalone by putting a request in, checking the transfer queue has a valid `KVTransfer`.

4. **Step 4: Implement `decode_worker()` (Hint 4)**
   - This is essentially your existing continuous batching loop, but sourcing requests from the transfer queue instead of doing its own prefill.
   - Test it standalone by manually constructing a `KVTransfer` and verifying it generates tokens.

5. **Step 5: Wire them together in `disaggregated_generate()` (Hint 5)**
   - Create the threads, feed requests, collect results.
   - Add telemetry logging (Hint 6).

6. **Step 6: Build the benchmark suite (Hints 7 & 8)**
   - Compare monolithic vs disaggregated: TTFT, total time, correctness.
   - Add staggered arrival scenarios.

7. **Step 7: Correctness verification (Test 1)**
   - Same seed, same requests → same output tokens.
   - This is the most important test — if the KV handoff is broken, tokens will diverge.

## Connection to Production Systems

| Your Implementation | Production (vLLM P/D) |
|----|-----|
| `threading.Thread` | Separate GPU processes |
| `queue.Queue` | NCCL / RDMA / TCP KV transfer |
| Shared `model` object | Separate model replicas on each GPU |
| `KVTransfer` dataclass | KV cache transfer protocol (tensor metadata + data) |
| `clone()` KV tensors | GPU-to-GPU memory copy |
| Single-machine overlap | Multi-node pipeline parallelism |

The architectural pattern is identical — you're just simulating the network boundary with a thread-safe queue.
