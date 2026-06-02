# GPU–CPU Overlap in LLM Inference: From First Principles

---

# 1. The Core Insight: GPUs and CPUs Are Independent Processors

A modern inference server has two processors running simultaneously:

- **GPU**: Runs matrix multiplies, attention, and all heavy tensor operations
- **CPU**: Runs the inference framework (Python/C++), manages memory, handles network I/O, runs the scheduler, does tokenization/detokenization, manages the request queue

In a **naive implementation**, these operate sequentially:

```
CPU: receive request → prepare tensors → [idle while GPU works]
GPU: [idle while CPU prepares]  → forward pass → [idle while CPU processes output]
CPU: [idle]                     → decode token → send to client → prepare next step
```

The CPU and GPU take turns. Neither is fully utilized.

**GPU–CPU overlap** means running CPU work and GPU work **at the same time**, so neither waits for the other.

---

# 2. What Work Can Be Overlapped?

### CPU-side work during inference:

| Task | When it happens | CPU-bound? |
|---|---|---|
| **Tokenization** | Before prefill, for new requests | Yes |
| **Detokenization** | After each decode step, to stream text | Yes |
| **Sampling** (top-k, top-p, temperature) | After GPU produces logits | Partially (can be GPU or CPU) |
| **Scheduler decisions** | Between decode steps — which sequences to preempt, add, or evict | Yes |
| **KV cache management** | Block allocation, page table updates, swap scheduling | Yes |
| **Network I/O** | Receiving new requests, streaming tokens to clients (SSE/WebSocket) | Yes |
| **Request preprocessing** | Prompt parsing, chat template application, input validation | Yes |

### GPU-side work:

| Task | When it happens |
|---|---|
| **Prefill forward pass** | When processing a new prompt |
| **Decode forward pass** | Each token generation step |
| **KV cache memory operations** | Block copies, swaps, quantization |

### The overlap opportunity:

While the GPU executes the forward pass for decode step N:

```
GPU: [running forward pass for step N — takes ~30ms]
CPU: [simultaneously doing ALL of this]:
  - Detokenize token from step N-1 and send to client
  - Run scheduler: decide if new requests should join the batch
  - Tokenize any newly arrived requests
  - Update KV cache page tables
  - Handle network I/O (receive new requests, send SSE chunks)
  - Prepare input tensors for step N+1
```

When the GPU finishes step N, the CPU has already prepared everything for step N+1. The GPU can immediately start the next step with zero idle time.

---

# 3. How Overlap Is Implemented

### 3.1 CUDA Streams and Asynchronous Execution

CUDA operations are **asynchronous by default** when launched on a stream. The CPU issues a kernel launch and **immediately returns** to do other work. The GPU executes the kernel independently.

```python
# Pseudocode
gpu_stream = cuda.Stream()

# CPU launches GPU work (returns immediately)
gpu_stream.launch_kernel(forward_pass, input_tensor)

# CPU does its own work while GPU is busy
new_text = tokenizer.decode(last_token)
send_to_client(new_text)
scheduler.update_batch()
new_requests = network.poll()
for req in new_requests:
    tokens = tokenizer.encode(req.prompt)
    scheduler.add_request(tokens)

# Only synchronize when CPU needs the GPU result
gpu_stream.synchronize()  # blocks until GPU is done
logits = gpu_stream.get_result()
next_token = sample(logits)
```

The key: `launch_kernel` is non-blocking. The CPU continues executing Python/C++ code while the GPU runs the forward pass.

### 3.2 Double-Buffered Input Preparation

To avoid any synchronization stall, systems use two input buffers:

```
Buffer A: GPU is computing on this (step N)
Buffer B: CPU is preparing this (step N+1's inputs)

After GPU finishes step N:
  - Swap A and B
  - GPU starts step N+1 on what was buffer B (now ready)
  - CPU starts preparing step N+2 in what was buffer A (now free)
```

This ensures the GPU never waits for input preparation.

### 3.3 Asynchronous Networking

Token streaming to clients uses non-blocking I/O:

```python
# asyncio-based server (like FastAPI with uvicorn)
async def generate_stream(request):
    async for token in inference_engine.generate():
        yield f"data: {token}\n\n"  # SSE format
```

The network sends happen on a separate thread/event loop from inference. The GPU forward pass, CPU scheduling, and network I/O all run concurrently.

---

# 4. Concrete Example: Decode Loop With Full Overlap

A production-quality decode loop (simplified):

```
Step N:
  t=0ms     GPU: start forward pass (decode step N)
  t=0ms     CPU: detokenize token N-1, send SSE chunk to client
  t=1ms     CPU: scheduler checks for new requests, preemptions
  t=2ms     CPU: tokenize 2 new requests that arrived
  t=3ms     CPU: update page tables for KV cache blocks
  t=4ms     CPU: prepare input tensor for step N+1 (in buffer B)
  t=5-25ms  CPU: idle (waiting for GPU)
  t=30ms    GPU: forward pass complete, logits ready
  t=30ms    CPU: sample next token from logits
  t=30.5ms  GPU: start forward pass (decode step N+1, using buffer B)
  t=30.5ms  CPU: detokenize token N, send SSE chunk...
  ...repeat
```

Without overlap, the same cycle would take ~35ms (30ms GPU + 5ms CPU serial). With overlap: 30ms — the CPU work is completely hidden.

---

# 5. Where Overlap Breaks Down

### 5.1 Synchronization Points

Certain operations require the CPU to wait for the GPU (or vice versa):

- **Sampling**: The CPU needs the logits tensor (on GPU) to sample the next token. This requires a device-to-host copy + synchronization.
- **KV cache swap decisions**: The scheduler may need to know current GPU memory state before deciding whether to preempt.
- **Dynamic shape changes**: When sequences join or leave the batch, input tensor shapes change and the CPU must rebuild them with fresh GPU memory pointers.

Minimizing these sync points is critical. Some systems move sampling to the GPU to avoid the logits copy entirely.

### 5.2 Python GIL

In Python-based inference frameworks (most of them), the Global Interpreter Lock (GIL) prevents true CPU thread parallelism. Workarounds:
- Use **C++ extensions** for performance-critical CPU work (tokenization, scheduling)
- Use **multiprocessing** (separate processes for scheduler vs. network vs. inference)
- Use **asyncio** for network I/O (non-blocking, single-thread concurrency)

vLLM, SGLang, and TensorRT-LLM all use C++ for their hot paths to avoid GIL bottlenecks.

### 5.3 CPU Becoming the Bottleneck

In high-QPS scenarios with hundreds of concurrent requests, the CPU scheduler itself can become the bottleneck:
- Page table management for PagedAttention
- Tokenization/detokenization for many concurrent streams
- Request routing and load balancing

This is increasingly common as GPUs get faster (more tokens/sec) but CPU scheduling logic stays roughly the same speed.

---

# 6. The Future of GPU–CPU Overlap

### 1. GPU-Side Sampling and Post-Processing
Moving the sampling step (top-k, top-p, temperature, repetition penalty) entirely to the GPU eliminates one of the main CPU↔GPU synchronization points. Some frameworks already do this. The next step is GPU-side detokenization for direct-to-network streaming.

### 2. Dedicated Scheduling Coprocessors
As scheduling complexity grows (disaggregated serving, speculative decoding, dynamic KV cache management), the CPU scheduler becomes a bottleneck. Future systems may use dedicated lightweight processors (embedded ARM cores, FPGAs) for scheduling, freeing the host CPU entirely.

### 3. Grace Hopper Unified Memory
NVIDIA's Grace Hopper architecture allows CPU and GPU to share a unified memory address space with cache-coherent access. This eliminates explicit memory copies for scheduling metadata, page tables, and small tensors — fundamentally reducing synchronization overhead.

---

# 7. The Investor Lens (Aligned with the Inference Framework)

GPU–CPU overlap sits in the **Serving / Runtime Layer**. It is pure systems engineering that determines how much of the hardware's theoretical performance is actually delivered to users.

### Value Drivers

- **The hidden utilization gap**: A GPU with 3 TB/s of HBM bandwidth and 1000 TFLOPS of compute can still deliver poor tokens/sec if the CPU side introduces stalls after every decode step. GPU–CPU overlap closes this gap. The difference between 85% and 95% GPU utilization at scale is millions of dollars per year in hardware costs.
- **Python as a structural disadvantage**: The Python GIL creates a ceiling on CPU-side throughput. Inference frameworks that have rewritten their hot paths in C++/Rust (vLLM's engine, TensorRT-LLM) have a structural performance advantage. This is a meaningful differentiator for serving-layer companies.
- **Grace Hopper as an architectural bet**: The unified CPU–GPU memory model in Grace Hopper is NVIDIA's hardware-level solution to the overlap problem. If it delivers on its promise (zero-copy scheduling metadata, coherent page table access), it significantly devalues the engineering effort that inference frameworks have spent on manual overlap scheduling. Watch for benchmark disclosures comparing Grace Hopper vs. discrete GPU+CPU setups.

### Summary Signal

> GPU–CPU overlap is the "last 20% of performance" that separates research prototypes from production systems. Companies (and open-source projects) that master it extract materially more value from the same hardware. For investors, the signal is simple: measure tokens/sec per GPU-dollar across providers — the ones with highest utilization have the best overlap engineering, and that translates directly to margin.
