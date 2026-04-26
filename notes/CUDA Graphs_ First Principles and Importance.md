---
title: "CUDA Graphs: First Principles and Importance"
source: "https://grok.com/c/58be43b9-5307-4f89-8946-034cb46a30f1?rid=07ab38b7-4b21-4b2d-84eb-5e1199332cfa"
author:
published:
created: 2026-04-25
description: "Grok is an AI assistant built by xAI. Chat, create images, write code, and get real-time answers from the web and X."
tags:
  - "clippings"
---
explain cuda graphs from first principles and why they are important.

CUDA Graphs are a powerful feature in NVIDIA's CUDA programming model (introduced in CUDA 10) that allow you to define a sequence of GPU operations—such as kernel launches, memory copies, memory sets, host functions, and even sub-graphs—as a single reusable **directed acyclic graph (DAG)**. This graph can then be launched repeatedly with minimal overhead.

### From First Principles: Traditional CUDA Execution Model

To understand why graphs matter, start with the basics of how CUDA normally works:

1. **Host (CPU) vs. Device (GPU)**: You write code that runs on the CPU (host). To use the GPU (device), you launch **kernels** —functions that execute in parallel across thousands of GPU threads. You also perform data movements (e.g., cudaMemcpy) and other operations.
2. **Streams and Asynchronous Submission**: Most CUDA work is submitted asynchronously to **streams** (queues). Operations in the same stream execute in order, but the CPU doesn't wait for the GPU unless you explicitly synchronize. This allows overlap between CPU work, data transfers, and GPU compute.
3. **Kernel Launch Process**: Every time you launch a kernel (e.g., myKernel<<<grid, block>>>(args)), the CUDA runtime/driver on the CPU must:
	- Validate arguments.
		- Set up launch parameters (grid/block dimensions, shared memory, etc.).
		- Allocate/prepare internal resources.
		- Communicate the launch command to the GPU (via the driver and hardware queues).
		- Handle any dependencies via streams or events.
	This **launch overhead** is typically on the order of **3–10 microseconds** per launch on modern systems (exact numbers vary by GPU, driver, and workload).

When your kernels are **long-running** (e.g., milliseconds), this overhead is negligible—GPU compute dominates. But modern workloads (especially in deep learning, inference, or fine-grained parallel algorithms) often involve **many short kernels** (microseconds to low milliseconds) chained together with dependencies. Here, the CPU repeatedly pays this overhead, creating "gaps" where the GPU sits idle waiting for the next launch command. The CPU becomes a bottleneck, reducing GPU utilization.

In essence, traditional streams submit work **piecewise**: the CPU issues one operation at a time, incurring repeated setup costs.

### What Is a CUDA Graph?

A CUDA Graph flips this model:

- **Nodes**: Each node represents one asynchronous CUDA operation (kernel launch, cudaMemcpy, host callback, memory allocation, etc.).
- **Edges**: Explicit dependencies between nodes. A node only starts after all its predecessor nodes complete. This forms a DAG—no cycles.
- **Definition Phase**: You build the graph structure **once**, either:
	- **Manually**: Using APIs to add nodes and dependencies (cudaGraphAddKernelNode, etc.).
		- **Stream Capture** (most common and easiest): Put a stream into "capture mode," run your normal CUDA code (launches, copies, etc.), and the runtime records everything into a graph instead of executing it immediately. Then end capture.
- **Instantiation Phase**: Convert the graph into an **executable graph** (cudaGraphExec\_t). This "bakes in" much of the setup work, optimizes the dependency structure, and prepares hardware-level representations. It can involve uploading the graph to the GPU.
- **Execution Phase**: Launch the entire graph with a **single** CPU call (cudaGraphLaunch). The GPU executes the whole workflow according to the pre-defined dependencies, with minimal further CPU involvement.

You can launch the same instantiated graph **many times** (replay). For repeated workloads (common in training/inference loops), the expensive setup happens only once.

Graphs support:

- Multiple streams' worth of work (concurrency where possible).
- Hierarchical graphs (sub-graphs as nodes).
- Multi-GPU scenarios.
- Updates for limited dynamism (e.g., changing kernel parameters without full recapture, though full dynamism is limited).

### Why CUDA Graphs Are Important: Performance Benefits

The core value comes from **amortizing overhead** and enabling **higher-level optimizations**:

1. **Dramatic Reduction in Launch Overhead**:
	- Traditional: Pay ~5μs *per kernel* for every iteration of your loop.
		- Graphs: Pay the full setup cost **once** during definition/instantiation. Each replay is essentially one launch for the entire sequence.
		- Result: For workloads with dozens or hundreds of small kernels per iteration (e.g., neural network layers, iterative solvers), CPU-side costs drop sharply. GPU utilization rises because the GPU spends less time idle waiting for the CPU.
2. **Better GPU Utilization and Reduced Jitter**:
	- The GPU receives a complete workflow upfront, allowing better internal scheduling, hardware dependency enforcement (sometimes bypassing stream/event overhead), and reduced variability in timing.
		- In deep learning frameworks like PyTorch, graphs eliminate overhead from Python/C++ dispatch layers, memory allocation checks, etc., on every iteration.
3. **Quantitative Gains**:
	- Speedups of 1.5–3x (or more) are common in latency-sensitive or small-batch inference scenarios (e.g., LLM serving with many short operations).
		- Kernel launch overhead can be reduced by up to ~95% in extreme cases.
		- Examples: 2.3x speedup in LLaMA-7B inference; noticeable gains in benchmarks with short kernels; improved performance in NAS Parallel Benchmarks or image processing pipelines.
		- Benefits are largest when kernels are short relative to launch cost, and the same workflow repeats many times (amortization works).
4. **Additional Optimizations**:
	- The runtime/driver sees the **entire dependency graph** at once, enabling optimizations impossible with piecewise stream submission (e.g., better concurrency or fused scheduling decisions).
		- Frees CPU resources for other tasks (lower power, better multi-threading).

**Trade-offs/Limitations**:

- Graphs assume **static** structure: same kernels, same grid/block sizes, same memory addresses (pointers are baked in). For new data, you update buffers in place before replay.
- Initial definition/instantiation has higher cost—so graphs shine in **repeated execution**, not one-off work.
- Limited dynamism (though updates or recapture can help). Not a replacement for kernel fusion (which reduces memory traffic by combining kernels into one).
- Debugging can be trickier; tools like Nsight help visualize graphs.

### Simple Mental Example

Imagine a loop that does: copy H2D → kernelA → kernelB → kernelC → copy D2H, repeated 1000 times.

- **Streams**: 5 launches per iteration × 1000 = 5000 launches → lots of ~5μs overhead gaps → GPU idle time.
- **Graph**: Capture once (records the 5 nodes + dependencies), instantiate once, then 1000 single graph launches → near-zero per-iteration launch cost → kernels run back-to-back.

In frameworks like PyTorch, this is often wrapped in a context manager for easy use (with warmup iterations first).

CUDA Graphs address a fundamental scaling issue as GPUs get faster and workloads get more fine-grained: the CPU-GPU submission interface becomes the bottleneck. By treating complex workflows as first-class reusable objects, they keep expensive GPU hardware saturated, which is critical for high-performance computing, AI training/inference, and scientific simulations. For many production workloads today, enabling CUDA Graphs (where applicable) is a low-effort, high-impact optimization.

For deeper dives, refer to the official CUDA Programming Guide section on graphs or NVIDIA's technical blogs on the topic.

explain fp8 kernels from first principles and how that relates to cuda graphs

**FP8 kernels** are specialized CUDA kernels (or library-accelerated operations) that perform arithmetic using the **8-bit floating-point (FP8)** data type instead of the traditional FP32 or FP16. They were introduced by NVIDIA starting with the Hopper architecture (H100, 2022) and are now standard on Ada, Blackwell, and later GPUs. From first principles, here’s why they exist and how they work.

### 1\. Floating-Point Representation — The Foundation

Any floating-point number in binary follows the IEEE-754 pattern:

$$
\text{value} = (-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times (1 + \text{mantissa})
$$
- **FP32** (single precision): 1 sign bit + 8 exponent bits (bias 127) + 23 mantissa bits → ~9 decimal digits of precision, huge dynamic range (~10⁻³⁸ to 10³⁸).
- **FP16** (half precision): 1 + 5 + 10 bits → only ~3–4 decimal digits, range ~10⁻⁴ to 10⁴ (or BF16 variant with larger range but same precision).
- **FP8** (8 bits total): two standardized formats (NVIDIA + industry):
	- **E4M3** (4 exponent bits, bias 7, 3 mantissa bits): Range ≈ 2⁻⁶ to ≈ 448, good precision near zero. Excellent for **weights and activations** (most DNN values cluster near zero).
		- **E5M2** (5 exponent bits, bias 15, 2 mantissa bits): Much larger range (≈ 2⁻¹⁴ to ≈ 57 344), fewer mantissa bits. Often used for **gradients** or cases needing wider range.

These formats are deliberately chosen so that the statistical distribution of values in a neural-network layer fits inside the 8-bit “bucket” with **acceptable rounding error**. The key insight: neural nets are extremely tolerant of noise in weights/activations (they were trained with stochastic gradient descent anyway). Quantization error from FP8 is usually <0.1–1% after proper scaling, which is far cheaper than the 4×–2× memory and bandwidth savings.

### 2\. Why FP8 Kernels Exist (The Performance Bottleneck)

Modern AI workloads (LLMs, diffusion models, etc.) are dominated by **matrix multiplies** (GEMM):

$$
\mathbf{C} = \mathbf{A} \times \mathbf{B} + \mathbf{D}
$$

In a transformer layer this happens hundreds of times per forward pass.

- Memory bandwidth is the #1 limiter: loading FP32 weights for a 70B model requires ~280 GB. FP8 shrinks this to ~70 GB.
- Compute throughput: GPUs have dedicated **Tensor Cores** (matrix engines). On Hopper/Blackwell these execute FP8×FP8 → FP16/FP32 accumulator **at 2×–4× the rate** of FP16×FP16 (depending on exact mode and GPU).

An **FP8 kernel** is simply a CUDA kernel (or a highly-optimized one inside cuBLAS/cuDNN/CUTLASS/TransformerEngine) that:

1. Loads input matrices as FP8 (using \_\_nv\_fp8 or vector types like \_\_nv\_fp8x4).
2. Feeds them directly into Tensor Core instructions (e.g., mma or wmma PTX with.fp8 suffix).
3. Applies **per-tensor or per-channel scaling factors** (small FP32 multipliers) before/after the multiply to keep values in the representable FP8 range. This is called **quantization-aware execution**.
4. Accumulates in higher precision (FP16 or FP32) to avoid overflow/underflow, then optionally casts back to FP8.

Result: same mathematical work, but **~2× lower memory traffic** and **much higher TOPS** (tera-operations per second). A single FP8 GEMM kernel can deliver 2–4× more math per watt and per second than its FP16 counterpart.

### 3\. From First Principles: Why the GPU Can Do This So Fast

Tensor Cores are hardware systolic arrays. They are **not** general-purpose ALUs — they are wired specifically for matrix tiles. FP8 shrinks the data that must move through the huge on-chip register file and crossbar, so the same hardware can process **twice as many elements per clock** without increasing power or die area. The driver/compiler also fuses the scaling multiplies into the Tensor Core pipeline, eliminating extra kernel passes.

### How FP8 Kernels Relate to CUDA Graphs

CUDA Graphs (explained in the previous answer) capture an entire **sequence** of CUDA operations (kernel launches, memory copies, etc.) into a single reusable DAG. FP8 kernels and CUDA Graphs are **highly synergistic** for exactly the reasons graphs were invented:

- **FP8 kernels are extremely fast**. A single FP8 GEMM on an H100 or B200 can finish in tens-to-hundreds of microseconds. → Traditional stream launch overhead (≈ 3–10 μs **per kernel**) now becomes a **larger fraction** of total time. The GPU finishes one layer and then sits idle waiting for the CPU to issue the next kernel launch.
- **LLM inference / training loops are repetitive sequences** of FP8 kernels (LayerNorm → QKV proj → attention → MLP, all in FP8). With CUDA Graphs you:
	1. Capture the **entire forward pass** once (including all the FP8 GEMMs, scaling, and memory ops).
		2. Instantiate the graph.
		3. Replay it with **one single CPU call** per iteration.
	The expensive per-kernel launch cost disappears. The GPU receives the full dependency graph up-front and runs the entire chain back-to-back with hardware-enforced dependencies and maximal Tensor Core utilization.
- **Quantitative win**: In production LLM serving (vLLM, TensorRT-LLM, PyTorch 2.0+ with torch.compile + graphs), enabling FP8 **plus** CUDA Graphs routinely gives **1.5–3× higher throughput** (tokens/sec) compared to FP16 streams, and even larger gains at small batch sizes or low latency. The FP8 kernels do the heavy math faster; the graph removes the CPU “tax” that would otherwise waste that speed.
- **Extra synergy points**:
	- FP8 scaling factors are usually static (or updated via graph node updates) → they can be baked into the captured graph.
		- Hierarchical graphs let you nest an FP8 sub-graph (e.g., one transformer layer) inside a larger graph.
		- In dynamic-shape scenarios you still capture a “template” graph and update input pointers/scales before each launch — still far cheaper than re-launching every kernel.

In short: **FP8 makes each individual kernel faster**; **CUDA Graphs make the sequence of kernels run with near-zero overhead**. Together they attack both sides of the performance equation — compute efficiency inside the kernel and submission efficiency across kernels — which is why virtually every high-performance LLM inference stack today (2026) ships with both enabled by default.

### Simple Mental Picture

Think of an LLM forward pass as 100 tiny FP8 “firecrackers” going off in sequence.

- Without graphs: you light each fuse individually from the CPU → lots of dead time between bangs.
- With graphs + FP8: you pre-build the whole string of firecrackers, light it once, and the whole chain explodes continuously at maximum speed.

That is why FP8 kernels and CUDA Graphs are almost always discussed together in modern GPU-accelerated AI.

explain prefetch piplines from first principles and how it connects to all of the concepts you explained above

**Prefetch pipelines** (also called **prefetch pipelining** or **software-pipelined prefetch**) are a core optimization technique in high-performance CUDA kernels and LLM inference engines. They hide the enormous latency of GPU global memory (HBM) by **overlapping data loading for the “next” piece of work with computation on the “current” piece** — exactly like an assembly line where one station loads parts while another assembles them.

### From First Principles: Why Prefetch Pipelines Exist

GPUs are **latency monsters**:

- A single global-memory load can take **hundreds of clock cycles** (200–600+ on Hopper/Blackwell).
- Tensor Cores (the matrix engines used in FP8 kernels) can perform **thousands of FP8 operations per cycle** once the data is in registers or shared memory.
- Without overlap, the SM (Streaming Multiprocessor) spends most of its time **stalled** waiting for data → low utilization, even if your kernel is “fast.”

The classic solution inside a single kernel is **tiling + double buffering**:

1. Break the work (e.g., a GEMM) into small tiles that fit in fast on-chip memory (shared memory or registers).
2. Use **two (or more) buffers** (“ping-pong”):
	- Buffer A: compute on it now.
		- Buffer B: asynchronously prefetch the *next* tile into it.
3. When the current tile finishes, swap buffers and repeat.

This creates a **pipeline with stages**:

- **Stage 1 (Producer)**: Issue async loads / TMA copies for future data.
- **Stage 2 (Consumer)**: Do the heavy math (FP8 matmul, etc.) on already-loaded data.

Modern CUDA gives you hardware primitives to make this clean and safe:

- cuda::pipeline + \_\_pipeline\_memcpy\_async (since CUDA 11.4) — a high-level API for multi-stage producer-consumer patterns inside a thread block.
- Tensor Memory Accelerator (**TMA**, Hopper+) — dedicated hardware that does asynchronous 1D–5D tensor copies from global → shared memory, with **descriptor prefetching** (you can prefetch the TMA descriptor itself so the next load starts instantly).
- Warp specialization / thread-block clusters — some warps act as “loaders” (prefetch), others as “computers” (matmul), coordinated with barriers or DSMEM.
- memcpy\_async + cuda::barrier for simpler double/triple buffering.

The result: memory latency is **hidden behind useful compute**. The pipeline depth (number of stages/buffers) is chosen so that prefetch time ≈ compute time → zero stalls.

### Prefetch Pipelines in Practice for LLM Inference

In transformer layers the bottleneck is usually **weight loading** between layers (not just inside one GEMM):

- A 70B model has ~hundreds of GB of weights.
- In autoregressive decoding you process **one layer at a time** (QKV proj → attention → MLP → …).
- Weights are static but live in slow HBM; each layer’s weights must be loaded into registers/shared memory before the FP8 GEMMs run.

**Prefetch pipeline** does this across layers:

- While the current layer’s FP8 kernels are crunching activations,
- A background pipeline (async copies or dedicated loader warps / TMA) **streams the next layer’s weights** into on-chip memory.

This is often called **inter-layer prefetch pipelining** or simply “prefetch pipelines” in LLM papers and inference engines (vLLM, TensorRT-LLM, SGLang, etc.).

It is usually combined with:

- Persistent kernels (one big kernel that loops over many layers internally, keeping the pipeline alive).
- Or multi-kernel pipelines captured in **CUDA Graphs**.

### How Prefetch Pipelines Connect to Everything Explained Earlier

1. **Connection to FP8 Kernels** FP8 kernels are *blazingly fast* (2–4× the throughput of FP16 because of smaller data + denser Tensor Core ops). But speed exposes the next bottleneck: **memory stalls become a larger fraction of total time**. A pure FP8 GEMM without prefetch can easily drop to 30–50% utilization because the Tensor Cores finish a tile before the next tile arrives. Prefetch pipelines keep the FP8 engines **continuously fed**, turning a fast kernel into a *saturated* one. TMA + FP8 is especially powerful — NVIDIA designed TMA descriptors to be prefetchable exactly for FP8/INT8 GEMMs.
2. **Connection to CUDA Graphs** CUDA Graphs shine when you have a **repetitive, static sequence** of operations. A real LLM forward pass is exactly that: Layer 1 (prefetch Layer 2 weights) → FP8 GEMMs → Layer 2 (prefetch Layer 3 weights) → … Without graphs you pay kernel-launch overhead *between every layer*, which breaks the pipeline timing and adds jitter. With graphs you:
	- Capture the entire multi-layer pipeline (including all async prefetch copies and dependencies) **once**.
		- Instantiate once.
		- Replay with **one single CPU call** per token/step. The GPU hardware now sees the full dependency DAG and can schedule prefetch → compute overlaps perfectly, with zero host intervention. This is why production stacks (PyTorch 2.x + torch.compile, TensorRT-LLM, etc.) combine FP8 + graphs + prefetch pipelines for 1.5–3×+ gains in tokens/sec.
3. **The Full Synergy Picture**
	- **FP8 kernels** = make each math operation 2× cheaper and faster.
		- **Prefetch pipelines** = hide the memory cost *inside and between* those fast kernels.
		- **CUDA Graphs** = remove the CPU-side orchestration tax so the entire pipelined workflow runs as one atomic, hardware-scheduled unit.
	Together they attack the three main scaling walls in modern AI:
	- Arithmetic intensity (FP8),
		- Memory latency (prefetch pipelines),
		- Submission/launch overhead (graphs).

**Simple Mental Model** Think of an LLM layer as a factory:

- Without prefetch: Workers (Tensor Cores) finish assembling a car and then stand idle waiting for the next crate of parts from the warehouse (HBM).
- With prefetch pipelines: A conveyor belt (TMA + async copies) is already delivering the *next* crate while the workers finish the current car.
- With CUDA Graphs: The entire factory floor (all layers + conveyors) is pre-programmed as one reusable machine; you just press “start” once per token instead of micromanaging every station.

In 2026-era real-time Gen AI (low-latency inference, high tokens/sec), these three techniques are almost always used together. You’ll see them listed side-by-side in every serious optimization guide precisely because they multiply each other’s benefits.

explain gpu and cpu overlap using first principles and how it relates to all the other concepts above

**GPU-CPU overlap** (also called **CPU-GPU overlap** or **asynchronous execution**) is the fundamental technique of making the CPU and GPU work **at the same time**, instead of taking turns. From first principles, this is essential because the two processors are radically different and connected by a relatively slow "bridge."

### First Principles: Why CPU and GPU Need to Overlap

- **CPU strengths**: Excellent at sequential logic, control flow, branching, launching work, managing memory, Python/C++ orchestration, scheduling, etc. It is **slow at massive parallel math** (only a few dozen cores).
- **GPU strengths**: Thousands of cores optimized for data-parallel math (matrix multiplies, convolutions, etc.). It is **extremely fast at compute** once data is ready, but weak at complex control flow and has **very high latency** for global memory accesses.
- **The bridge**: Data lives in **separate memory spaces** (host RAM vs. device HBM). Moving data across PCIe (or NVLink) takes real time — often comparable to or longer than short GPU kernels. The CPU must issue commands (kernel launches, memory copies) to the GPU via the driver.

Without overlap:

- CPU issues work → waits (synchronizes) → GPU works → CPU waits again. This creates **serialization**: one sits idle while the other works. Utilization drops dramatically.

With overlap:

- CPU prepares the *next* piece of work (or does other useful tasks) **while** the GPU is still crunching the *current* piece.
- GPU starts the *next* kernel or data transfer **while** the CPU is still issuing commands or doing host-side preprocessing.

This is like a two-stage assembly line: one worker (CPU) prepares parts while the other (GPU) assembles them. The goal is to keep **both** busy as much as possible, hiding latencies.

### How CUDA Enables GPU-CPU Overlap (Basic Mechanisms)

1. **Asynchronous Operations**:
	- Kernel launches (<<<>>>) and cudaMemcpyAsync return **immediately** to the CPU. The work is queued on the GPU.
		- The CPU can continue executing code right away.
2. **Streams**:
	- Streams are independent queues. Operations in different streams can run concurrently (copy in one stream while computing in another).
		- This enables classic **copy-compute overlap**: while one batch’s data is transferring H2D, another batch’s kernel runs on the GPU.
3. **Events and Synchronization**:
	- You only synchronize (cudaStreamSynchronize or events) when you *must* wait for results. Smart code minimizes these points.
4. **Pinned (Page-Locked) Host Memory**:
	- Allows true asynchronous DMA transfers that overlap with GPU compute (without it, the driver may block).

Result in a timeline view (Nsight Systems style):

- Good overlap: CPU bars and GPU bars run in parallel with minimal white space (idle time).
- Poor overlap: Long stretches where only CPU or only GPU is active.

### How GPU-CPU Overlap Relates to All the Previous Concepts

These techniques build on each other in a beautiful hierarchy. Each layer attacks a different source of idle time:

1. **Connection to Traditional Streams (the foundation)** Basic overlap relies on streams for concurrent copy + compute or multi-kernel overlap. Without streams, everything serializes in the default stream. This is the entry-level way to hide H2D/D2H transfers behind GPU kernels or CPU work.
2. **Connection to CUDA Graphs** Graphs take overlap to the next level for **repetitive workloads** (exactly what LLM inference is).
	- In plain streams: For each iteration you have a long chain of small operations (many FP8 GEMMs, LayerNorms, etc.). The CPU must repeatedly issue launches, incurring **driver + runtime overhead** (~3–10 μs each). This overhead itself becomes a bottleneck — the CPU spends too much time "talking" and can't prepare the next work fast enough, or the GPU finishes early and idles waiting for the next launch command.
		- With graphs: You capture the **entire sequence** once. Then each replay is **one single launch**. The CPU overhead drops dramatically, so the CPU stays free to do other useful work (e.g., Python orchestration, dynamic batching, sampling, KV cache management, or even overlapping with another GPU's work in multi-GPU setups). The GPU receives a complete, pre-baked DAG and can execute with maximal internal concurrency and minimal jitter. Graphs essentially **amplify** overlap by removing the CPU as a frequent interrupter.
3. **Connection to FP8 Kernels** FP8 kernels make each individual operation **much faster** (higher TOPS, lower memory traffic).
	- Faster kernels = shorter compute phases → the relative cost of CPU launch overhead and memory stalls **increases**.
		- Without excellent overlap, the GPU finishes an FP8 layer in microseconds and then sits idle waiting for the CPU to issue the next tiny kernel.
		- Good CPU-GPU overlap (enabled by graphs + streams) ensures that as soon as one FP8-heavy layer finishes, the next is already queued and ready. FP8 exposes the need for overlap; good overlap lets you actually realize FP8’s speed.
4. **Connection to Prefetch Pipelines** Prefetch pipelines are themselves a form of **internal GPU overlap** (memory latency hidden behind compute via async loads/TMA + double/triple buffering).
	- But they connect to CPU-GPU overlap too: While the GPU is computing the current layer (using prefetched weights/activations), the CPU can be preparing the *next* input tokens, updating KV cache, or even issuing prefetch commands for the following layer’s weights from CPU RAM or disk.
		- In advanced LLM engines, you often see **multi-level pipelining**: GPU-internal prefetch (TMA) + inter-layer weight prefetch + CPU-side orchestration, all overlapped via graphs and streams. The result is a deep pipeline where CPU work, data movement, and GPU compute all run concurrently across layers and batches.

### The Full Synergistic Picture in Modern LLM Inference (2026)

A high-performance LLM forward pass today looks like this:

- **CPU** (orchestration thread): Dynamically batches requests, manages KV cache, does sampling/logits processing, prepares next inputs — all while the GPU is busy.
- **GPU** (via CUDA Graph replay): Executes the full layer pipeline (prefetch next weights → FP8 QKV proj → attention → FP8 MLP → …) with internal prefetch pipelines keeping Tensor Cores fed. One graph launch covers dozens of layers with near-zero per-layer CPU involvement.
- **Overlap everywhere**:
	- CPU preparing batch N+1 while GPU runs batch N.
		- H2D transfers (new tokens) overlapping with GPU compute on previous work.
		- Inter-layer weight prefetch overlapping current-layer math.
		- Multiple streams or graph-internal concurrency allowing copy/compute overlap even inside the captured graph.

This combination routinely delivers **1.5–4× higher throughput** (tokens/second) compared to naive synchronous code. The GPU stays saturated (>90% SM utilization), the CPU is not the bottleneck, and latencies are predictable.

**Simple Mental Model** Imagine building a car on a moving assembly line:

- Without overlap: Stop the line, CPU brings parts, GPU assembles one piece, repeat → slow and jerky.
- With basic streams: Line keeps moving; CPU brings next parts while GPU assembles.
- With prefetch pipelines: Parts are pre-loaded onto the line ahead of time.
- With FP8: The assembly robots work twice as fast on smaller, denser parts.
- With CUDA Graphs: The entire factory line (all stations) is pre-programmed as one reusable machine; you just hit “start” once per car instead of micromanaging every robot.

In real-time generative AI, keeping both CPU and GPU continuously useful is the difference between mediocre and state-of-the-art serving performance. All the concepts we’ve discussed — CUDA Graphs, FP8 kernels, and prefetch pipelines — are powerful individually, but they **multiply** when combined through excellent GPU-CPU overlap. This is why production inference frameworks treat them as a cohesive stack rather than isolated tricks.