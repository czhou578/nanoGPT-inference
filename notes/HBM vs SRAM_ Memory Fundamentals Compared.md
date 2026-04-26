---
title: "HBM vs SRAM: Memory Fundamentals Compared"
source: "https://grok.com/c/e0867f86-f7c6-4604-acd4-b13990f61633?rid=a5184fd4-bd07-46e8-8bd7-926e5b0f79b5"
author:
published:
created: 2026-04-25
description: "Grok is an AI assistant built by xAI. Chat, create images, write code, and get real-time answers from the web and X."
tags:
  - "clippings"
---
what is hbm vs sram from first principles?

**HBM (High Bandwidth Memory) and SRAM (Static Random Access Memory)** are two fundamentally different memory technologies in the computer memory hierarchy. They address different trade-offs in speed, capacity, power, cost, and physical integration. Here's a breakdown from first principles—starting with how bits are stored and accessed at the transistor/circuit level, then scaling up to architecture, performance, and real-world use (especially in GPUs and AI accelerators).

### 1\. How Data Is Stored: The Basic Cell (Bit-Level Physics)

Memory ultimately stores a **0** or **1** using electrical states in silicon transistors and/or capacitors.

- **SRAM (Static RAM)**:
	- Each bit uses a **6-transistor (6T) cell** in modern CMOS processes (the most common design).
		- Core: Two **cross-coupled inverters** (4 transistors: two pull-up PMOS and two pull-down NMOS) form a **bistable latch** (flip-flop). This latch holds its state indefinitely as long as power is supplied—it's "static."
		- Two additional **access transistors** (NMOS) connect the latch to bit lines when the word line is activated.
		- No capacitors involved. The state is actively maintained by the feedback loop between the inverters. It's stable against noise (within limits) and doesn't require periodic refreshing.
		- Why it works from physics: The inverters fight each other in a balanced way; one output high forces the other low, and vice versa. Reading/writing perturbs this balance temporarily but it snaps back.
- **HBM**:
	- HBM is **not** a new storage cell—it's **3D-stacked DRAM (Dynamic RAM)**.
		- Base cell: **1T1C** (1 transistor + 1 capacitor). The bit is stored as **electrical charge** in a tiny capacitor (0 = discharged, 1 = charged, or vice versa).
		- The transistor acts as a switch to access the capacitor.
		- **Dynamic** because the capacitor leaks charge over time (due to leakage currents in silicon). It must be **refreshed** every ~64 ms or so, which consumes power and adds overhead.
		- HBM stacks 8–16 (or more) of these DRAM dies vertically on a base logic die, connected by **Through-Silicon Vias (TSVs)** —tiny vertical wires drilled through the silicon.

**Key first-principles difference**: SRAM is a pure logic circuit (all transistors, no charge storage that leaks), while HBM/DRAM relies on analog charge storage that degrades. This makes SRAM inherently more reliable for speed but far less dense.

### 2\. Access and Operation

- **SRAM**:
	- Read: Precharge bit lines, activate word line → one bit line discharges based on the latch state. Very fast (picoseconds to low nanoseconds) because it's local, differential signaling, and no refresh.
		- Write: Drive bit lines to force the latch into the new state (stronger drivers overpower the feedback).
		- No refresh → predictable, low-latency access. Energy is mainly for switching transistors.
- **HBM (DRAM-based)**:
	- Read: Activate row (word line), sense amplifiers detect tiny voltage from the capacitor charge, then rewrite it back (destructive read).
		- Write: Similar, but involves charging/discharging the capacitor.
		- Refresh cycles steal bandwidth periodically.
		- The stacking + wide interface (1024+ bits wide per stack) allows massive parallelism: many bits transferred in parallel over short distances via TSVs.

SRAM access is **orders of magnitude faster in latency** and has **extremely high internal bandwidth** when on-chip because it's right next to logic. HBM trades some latency for vastly higher aggregate bandwidth through width and stacking.

### 3\. Scaling and Density (Why One Is Small and the Other Is "Large")

- **SRAM density**: Terrible. A 6T cell takes a lot of silicon area (roughly 5–6× less dense than a DRAM cell per mm² on the same process). On advanced nodes (e.g., TSMC N3), you might get ~38 Mb/mm² of SRAM. It's built on the same logic process as the CPU/GPU cores, so it's expensive and power-hungry per bit but integrates seamlessly.
- **HBM/DRAM density**: Excellent. 1T1C cells are tiny. Stacking 12–16 dies multiplies capacity dramatically (HBM stacks can hold tens of GB per package). Effective density per package is ~80× or more higher than equivalent SRAM area.

Result: SRAM is used in tiny quantities (MBs on a GPU die, e.g., ~20–200+ KB per streaming multiprocessor in NVIDIA GPUs, totaling ~20 MB total L1/shared memory). HBM provides **tens of GB** (up to 80+ GB on high-end GPUs) as "main" high-speed memory for the accelerator.

### 4\. Bandwidth, Latency, Power, and Cost

From real examples (e.g., NVIDIA A100/H100-class GPUs):

- **Bandwidth**:
	- On-chip SRAM: Extremely high—up to ~19 TB/s (internal to the chip, across many small banks close to compute units). This is because data paths are short, highly parallel, and low-capacitance.
		- HBM: Very high for off-chip memory—~1.5–2+ TB/s per GPU (HBM3/HBM3e), scaling to 2+ TB/s in HBM4. Achieved via ultra-wide bus (1024–2048+ bits) and high signaling rates, plus short interconnects via 2.5D interposer.
		- SRAM often wins by 5–10× in raw peak internal bandwidth for hot data.
- **Latency**:
	- SRAM: Lowest possible (on-chip, ~1–few cycles).
		- HBM: Much higher (tens to hundreds of cycles) due to off-chip distance, even if "close" via interposer.
- **Power/Efficiency**:
	- SRAM: Very efficient per access for on-chip data (no refresh, short wires). But static leakage can add up.
		- HBM: Better energy per bit transferred than older GDDR/DDR thanks to short TSV paths and wide interface, but refresh + longer distances make it more power-hungry overall for large capacities. Still optimized for high-performance computing.
- **Cost**:
	- SRAM: Extremely expensive per bit (100×+ vs. DRAM). Limited to caches/register files.
		- HBM: Expensive (3×+ per GB vs. standard DDR5) due to stacking, TSVs, and yield challenges, but far cheaper than equivalent SRAM capacity.

### 5\. Where They Fit in the Memory Hierarchy (and AI Context)

Computers use a **hierarchy** because no single tech is perfect:

- **Registers + L1 Cache (SRAM)** → Fastest, smallest (KB–MB), closest to ALUs/Tensor Cores.
- **L2/L3 (often SRAM or SRAM-like)** → Larger but slower.
- **HBM (in GPUs/accelerators)** → High-bandwidth "main memory" for model weights, activations, KV cache in LLMs. Sits on the package via silicon interposer.
- **System DRAM (DDR)** → Even larger/cheaper but slower.
- **Storage (SSD/HDD)** → Slowest.

In **AI/GPU workloads** (training/inference for large models):

- **SRAM** shines for "hot" data that must be accessed ultra-frequently and with minimal latency (e.g., tiling in FlashAttention: load small Q/K/V tiles into SRAM, compute entirely there to avoid slow HBM round-trips). Techniques like kernel fusion and tiling minimize HBM accesses.
- **HBM** is the workhorse for holding the massive model (weights often 10s–100s of GB) and providing TB/s bandwidth to feed thousands of compute units. It's the bottleneck breaker compared to older GDDR.
- Trade-off: SRAM is "insanely fast" but models that fit entirely in SRAM are ~100× smaller. HBM enables scale at the cost of some latency/bandwidth-per-access.

Recent discussions (e.g., around custom AI chips) explore larger on-chip SRAM arrays for ultra-low-latency inference (reducing data movement), while HBM dominates for high-capacity training/batched workloads.

### Summary Table (Approximate/Illustrative)

| Aspect | SRAM | HBM (Stacked DRAM) |
| --- | --- | --- |
| Storage per bit | 6 transistors (latch) | 1T + 1C (charge) + refresh |
| Density | Low (~38 Mb/mm²) | High (stacks → GBs per package) |
| Capacity (typical) | MBs total on-chip | 10s–100+ GB per GPU |
| Latency | Lowest (on-chip) | Higher (off-chip-ish) |
| Bandwidth | Highest internal (~10–20 TB/s) | Very high (~1–2+ TB/s) |
| Power per access | Excellent for hot data | Good, but refresh overhead |
| Cost per bit | Very high | High (but << SRAM for capacity) |
| Use case | Caches, registers, tiling | High-bandwidth main memory for AI |

**Bottom line from first principles**: SRAM wins on raw speed and simplicity (no refresh, logic-compatible) but loses badly on density/cost → use it for what must be fastest. HBM cleverly scales DRAM's density + adds massive parallelism via 3D stacking and wide interfaces to deliver "high bandwidth" where raw DRAM would fall short. They complement each other: SRAM feeds the compute cores from HBM in modern accelerators.

This hierarchy is why optimizations like FlashAttention exist—to keep as much work as possible in fast SRAM while streaming from HBM efficiently.

give me an indepth breakdown of gpu memory hierarchy from first principles.

**GPU Memory Hierarchy from First Principles**

The GPU memory hierarchy exists because of inescapable physics: **data movement costs energy and time**, while compute (ALUs, tensor cores) is getting cheaper and more parallel every process node. Light (and electrons) travel at finite speed, wires have resistance/capacitance (RC delays), and transistors occupy area. You cannot make every bit of memory simultaneously:

- infinitely fast (low latency),
- infinitely wide (high bandwidth),
- infinitely dense (high capacity),
- and infinitely cheap/power-efficient.

The hierarchy is a pyramid of trade-offs, engineered so that the **working set** of data stays as close to the compute units as possible (principle of locality: temporal + spatial). This minimizes average energy-per-bit-moved and latency while maximizing throughput. GPUs amplify this with **massive thread-level parallelism** (tens of thousands of threads) to hide latency by switching to other work.

Everything below is built from CMOS transistors on the same silicon die (except the outermost level). I'll break it down level-by-level, starting at the physics of the storage cell, then capacity/latency/bandwidth, access model, and why it exists. Numbers are illustrative for a modern high-end GPU (NVIDIA Hopper/Blackwell-class or AMD CDNA3-equivalent, ~2024–2026 era); exact figures vary by model but the ratios are what matter.

### 1\. Registers (Per-Thread Private Storage)

**Physics / Cell**: Pure SRAM. Each 32-bit register is a 6-transistor (6T) latch (two cross-coupled inverters + two access transistors). No capacitors, no refresh. The state is held by positive feedback in the inverters—electrons flow continuously to fight leakage.

- **Capacity**: Enormous per SM (Streaming Multiprocessor / Compute Unit). A single SM has ~256 KB of register file (65,536 × 32-bit registers). Across 100+ SMs: **tens of MB total**.
- **Latency**: ~1 clock cycle (sub-nanosecond on-chip). Shortest possible wires.
- **Bandwidth**: Insanely high— **tens of TB/s aggregate** across all SMs because every thread can read/write its own registers in parallel. Each warp (32 threads) can access 128–256 bytes/cycle easily.
- **Access model (CUDA perspective)**: Private to each thread. Compiler allocates automatically. No bank conflicts inside a thread, but register pressure can spill to L1 (rarely done in optimized kernels).
- **Power**: Lowest energy per access (~0.01–0.1 pJ/bit) because wires are microns long and capacitance is tiny.
- **Why it exists**: Hides latency of slower memory. GPUs run 10,000–100,000+ threads to keep tensor cores busy while waiting for data. Registers are the only level that scales with thread count.

**First-principles insight**: Registers are the “zero-distance” memory. Moving a bit farther costs ~10× more energy due to longer wires and larger drivers.

### 2\. L1 Cache + Shared Memory (Per-SM Scratchpad / Cache)

**Physics / Cell**: Still SRAM (6T cells), built on the same high-speed logic process as the cores. Physically co-located inside each SM.

- **Capacity**: Configurable 128–256 KB per SM (e.g., 192 KB default on Hopper). Can be split 50/50 or 0/100 between L1 cache and shared memory. Total across GPU: **~20–50 MB**.
- **Latency**: 1–few cycles (~few ns). Still on-die, next to the ALUs/tensor cores.
- **Bandwidth**: Extremely high— **multiple TB/s per SM** (e.g., 128 bytes/cycle per warp in shared memory). Banks are 32-wide (one per thread in a warp) to support full coalescing.
- **Access model**:
	- **Shared Memory**: User-controlled scratchpad (CUDA \_\_shared\_\_). Explicit load/store. Banked (32 banks); conflicts serialize if multiple threads hit same bank.
		- **L1 Cache**: Automatic hardware-managed cache for global memory loads. Weak coherence (GPU model is relaxed). Reads are cached; writes usually go through L2.
		- Texture/Constant caches are specialized read-only SRAM subsets here (optimized for 2D spatial locality or broadcast).
- **Power**: Still very low (~0.1 pJ/bit).
- **Why it exists**: **Tiling / data reuse**. In matrix multiply (GEMM, the heart of AI), you load a tile of A/B matrices into shared memory once, then every thread reuses it thousands of times in registers. This is the foundation of FlashAttention, cuBLAS, etc. Without it, every access would hit slower memory.

**First-principles insight**: Shared memory is “software-managed cache” because the programmer knows the access pattern better than hardware. It avoids cache thrashing in predictable workloads (AI, graphics).

### 3\. L2 Cache (Chip-Wide Unified Cache)

**Physics / Cell**: SRAM again (larger banks, still 6T). Larger transistors for density, but same principle.

- **Capacity**: 40–100+ MB on high-end GPUs (e.g., ~50 MB on H100, larger on B200). Shared across all SMs.
- **Latency**: 20–100+ cycles (tens of ns). Still on-die but farther away; crossbar interconnect adds delay.
- **Bandwidth**: High but lower than L1— **~5–10 TB/s aggregate** (serves all SMs in parallel via high-speed on-chip fabric).
- **Access model**: Automatic. Acts as a “victim cache” for L1 spills and a coherence point. Holds read-only data, textures, and some global data. Eviction policy is LRU-like.
- **Power**: Higher than L1 (~1 pJ/bit) due to longer wires and bigger drivers.
- **Why it exists**: Captures data that is shared across SMs (e.g., model weights reused by many thread blocks) without going off-chip. Reduces pressure on HBM.

**First-principles insight**: L2 is the last level that is still pure on-die SRAM. Beyond this, physics forces you to leave the compute die.

### 4\. Global Memory (HBM / GDDR – Off-Package High-Bandwidth DRAM)

**Physics / Cell**: DRAM (1T1C). Bit stored as charge in a capacitor; leaks → requires refresh (~64 ms). Stacked 8–16 dies high with Through-Silicon Vias (TSVs) + silicon interposer connecting to the GPU die. (GDDR is non-stacked variant used on consumer cards.)

- **Capacity**: Massive— **24–192 GB** per GPU (HBM3/HBM3e/HBM4). This is the “main memory” for the entire chip.
- **Latency**: 200–800+ cycles (hundreds of ns). Off-die, even with interposer (short ~mm traces).
- **Bandwidth**: Very high for off-chip— **1.5–3+ TB/s** per GPU (e.g., H100 = 3.35 TB/s with HBM3e). Achieved by ultra-wide interface (1024–2048 bits) + high clocking + many parallel stacks. HBM4 pushes even higher.
- **Access model**: Global (\_\_global\_\_ in CUDA). Coalesced accesses (128-byte sectors preferred). Atomic operations, etc. No per-thread private storage here.
- **Power**: Highest (~10–50 pJ/bit moved). Refresh + long wires + package capacitance.
- **Why it exists**: Only technology that gives you GB-scale capacity at reasonable cost. HBM’s 3D stacking + wide bus is the engineering miracle that makes modern AI training possible (a single GPU can hold a 70B model in FP16).

**First-principles insight**: HBM is DRAM scaled by parallelism (width + stacking) rather than speed. You pay the DRAM latency/refresh tax once per access, then amortize it over massive transfers.

### The Full Pyramid + Data Movement Costs

| Level | Technology | Typical Capacity (per GPU) | Latency (cycles) | Peak Bandwidth (aggregate) | Energy / bit | Primary Use |
| --- | --- | --- | --- | --- | --- | --- |
| Registers | SRAM (6T) | Tens of MB (thread-private) | 1 | 10s–100+ TB/s | ~0.01 pJ | Per-thread scalars, accumulators |
| L1 / Shared Mem | SRAM | 20–50 MB | 1–few | 10s of TB/s | ~0.1 pJ | Tiling, reuse in kernels |
| L2 Cache | SRAM | 40–100+ MB | 20–100 | ~5–10 TB/s | ~1 pJ | Cross-SM reuse |
| Global (HBM) | Stacked DRAM | 24–192 GB | 200–800+ | 1.5–3+ TB/s | 10–50 pJ | Model weights, activations, KV cache |

**Key ratios**:

- Latency multiplies by ~10× per level.
- Energy per bit multiplies by ~10× per level.
- Capacity multiplies by ~100–1000× per level.

**Why this exact structure?**

- **Roofline model**: Performance is bounded by arithmetic intensity (FLOPs per byte moved). High-intensity kernels (matrix multiply) live in registers/shared mem. Low-intensity kernels (scatters, reductions) are HBM-bound.
- **Power wall**: Data movement now dominates power budget (>50% in large models). Hierarchy + tiling (FlashAttention-2, etc.) keeps hot data close.
- **Amdahl’s law for memory**: Serial latency is hidden by massive parallelism; bandwidth is the real limiter.
- **Modern extensions** (Hopper+): Tensor Memory Accelerator (TMA) does asynchronous HBM↔shared copies without SM involvement. Asynchronous pipelines overlap compute and memory.

**Bottom line from first principles**: The GPU is a bandwidth monster because it puts **massive SRAM close to compute** (registers + shared) and **massive parallel DRAM one package away** (HBM). Every optimization in AI (kernel fusion, quantization, paging, FlashAttention) is ultimately about **reducing trips down the hierarchy**. The physics hasn’t changed since the first SRAM/DRAM cells were invented; we just got better at hiding the costs with parallelism and software smarts.

what is the difference between l1 and l2 cache and why is it named like that?

**L1 and L2 caches** are part of the multi-level cache hierarchy in both CPUs and GPUs. The naming comes from a simple, logical ordering: **Level 1 (L1)** is the **first** (closest and fastest) level of cache that the processor checks, **Level 2 (L2)** is the **second** level, and so on (L3 is third, etc.).

This "level" convention originated in computer architecture in the 1980s–1990s as processors added more cache layers to bridge the growing speed gap between fast logic (cores) and slower main memory (DRAM). L1 was the original on-chip cache; when designers added another larger but slower cache, they naturally called it L2. The numbers simply indicate proximity in the hierarchy and access order—not arbitrary labels.

### Core Differences: L1 vs L2 (General Principles + GPU-Specific)

Both are built from **SRAM** (6-transistor cells), but they optimize different trade-offs in the memory hierarchy:

- **Location and Proximity**:
	- **L1**: Private to each core (CPU) or each **Streaming Multiprocessor (SM)** / Compute Unit (GPU). Physically placed right next to the execution units (ALUs, tensor cores). Shortest wires → minimal RC delay.
		- **L2**: Shared across multiple cores/SMs. Located farther away on the die, often in a central area or via a crossbar interconnect.
- **Capacity**:
	- **L1**: Small — typically **tens to hundreds of KB** per core/SM. (On modern GPUs like Hopper/Blackwell: ~128–256 KB per SM, often configurable/shared with other structures.)
		- **L2**: Much larger — **tens of MB** total for the whole chip (e.g., 40–100+ MB on high-end GPUs). This allows it to capture data that misses in the many small L1s.
- **Speed / Latency**:
	- **L1**: Fastest cache access — usually **1–few cycles** on GPUs (or ~20–40 cycles in some profiled GPU numbers, still far faster than further levels). Extremely low latency because of proximity.
		- **L2**: Slower — **tens to hundreds of cycles** on GPUs (e.g., ~150–700+ cycles depending on distance/hit type). Still much faster than HBM/global memory (~200–800+ cycles).
- **Bandwidth**:
	- **L1**: Extremely high per SM/core (multiple TB/s aggregate across the chip) because many parallel accesses happen locally.
		- **L2**: High but lower per-access bandwidth than L1; it serves the entire chip, so contention is higher.
- **Management and Access Model** (especially on GPUs):
	- **L1**: On NVIDIA GPUs, the L1 data cache is often **unified** with **shared memory** in the same SRAM block. You can configure the split (more shared memory for explicit tiling, or more L1 for automatic caching of global loads). L1 is optimized for **warp-level coalesced accesses** and has larger cache lines in some generations (e.g., 128-byte for L1 hits vs. 32-byte sectors for L2-only in older NVIDIA designs). This reduces over-fetch for scattered patterns when bypassing L1.
		- **L2**: Purely hardware-managed, unified across all SMs. It acts as a "victim cache" or backing store for L1 misses. It helps with data reuse **across different SMs/thread blocks** (e.g., model weights accessed by many parts of the GPU). On GPUs, L2 reduces traffic to HBM and handles coherence-like functions in a relaxed GPU memory model.
- **Purpose in the Hierarchy**:
	- **L1** filters the hottest, most frequently accessed data for its local execution unit. It exploits **temporal locality** (reuse soon) and **spatial locality** (nearby data) at the finest grain.
		- **L2** catches misses from all the L1s, providing a larger "safety net" without the area/power cost of making every L1 huge. It improves overall hit rate for data shared across the chip.

In GPUs specifically:

- L1 (per-SM) shines for **tiling** in compute-heavy kernels like matrix multiplication or FlashAttention — load a tile once into L1/shared, reuse it heavily in registers.
- L2 helps when data is reused across SMs but doesn't fit in individual L1s, or for irregular/scattered accesses that benefit from larger capacity.

### Why Multiple Levels Exist (First-Principles Reasoning)

You cannot optimize one cache for everything due to physics:

- Smaller + closer = faster + lower power (shorter wires, less capacitance) but limited capacity → high risk of thrashing if too big.
- Larger capacity requires more silicon area, longer wires/search times, and more power for tags/drivers → slower.

L1 is tiny and ultra-fast to match core speed. L2 is bigger/slower to absorb more misses without going all the way to DRAM/HBM. This multi-level approach dramatically improves **average memory access time** (AMAT) while keeping power and area reasonable. Adding L3 (on CPUs, sometimes large last-level cache) follows the same logic.

On GPUs, the hierarchy is even more pronounced because of massive parallelism: thousands of threads hide latency by switching, but bandwidth and data movement still dominate energy use. Optimizations (coalescing, tiling, asynchronous copies via TMA) all aim to maximize hits in L1 and minimize expensive trips to L2 or HBM.

### Quick Comparison Table (GPU-Focused, Approximate for Modern High-End)

| Aspect | L1 Cache (per SM) | L2 Cache (chip-wide) |
| --- | --- | --- |
| Location | Inside each SM, next to cores | Shared, farther on die |
| Capacity | 128–256 KB per SM | 40–100+ MB total |
| Latency | Very low (few–dozens of cycles) | Higher (100–700+ cycles) |
| Bandwidth | Highest local (TB/s per SM) | High but shared |
| Management | Often configurable with shared mem | Fully hardware-managed |
| Best for | Per-block tiling & local reuse | Cross-SM reuse & reducing HBM traffic |

In short: **L1 is the "first responder"** — smallest, fastest, private. **L2 is the "backup"** — larger, slower, shared. The numbering simply reflects the order in which the hardware checks them on a miss (L1 first → L2 → HBM/global memory).

This design has remained remarkably consistent since early cache hierarchies because it elegantly solves the fundamental speed-density-power trade-offs in silicon.

connect this to batching in llm inference from first principles and why.

**Batching in LLM inference is the direct, physics-driven consequence of the GPU memory hierarchy we’ve been discussing.** It exists because data movement (especially from HBM) is 100–1000× more expensive than compute, and the hierarchy is engineered precisely to keep the working set as close to the tensor cores as possible. Batching is the software technique that maximizes **arithmetic intensity** (FLOPs per byte moved from slow memory) so the GPU spends more time computing in registers/L1 and less time waiting on HBM → L2 → L1 round-trips.

Let’s connect it rigorously from first principles, starting at the transistor level and working up.

### 1\. First-Principles Bottleneck in LLM Inference

A transformer layer (the core of every LLM) consists of:

- **Matrix multiplies** (QKV projection, attention scores, FFN) — each is a GEMM: C = A × B.
- **Attention** — involves loading/storing the **KV cache** (key and value tensors that grow with sequence length).
- Autoregressive generation: one token at a time, but the model weights (~tens to hundreds of GB in FP16/BF16) must be read **every single token**.

Physics reminder from the hierarchy:

- HBM (global memory) access: ~10–50 pJ/bit + 200–800+ cycles latency.
- L2: ~1 pJ/bit, 100–700 cycles.
- L1 / Shared memory: ~0.1 pJ/bit, few cycles.
- Registers: ~0.01 pJ/bit, 1 cycle.

In a single forward pass, the **model weights** live in HBM (they are far too big for L2 or L1). Every GEMM therefore requires:

- Load weights from HBM → stream through L2 → L1/Shared → registers.
- Load activations/KV from HBM (or page them).
- Do massive FLOPs inside the tensor cores (which are fed from registers).

For **batch size = 1** (one sequence):

- You load ~same amount of weights per token.
- You perform relatively few FLOPs per token (especially in early layers).
- Result: **extremely low arithmetic intensity** → the GPU is **memory-bound**. Most time is spent on HBM → L2 → L1 traffic. Tensor cores sit idle waiting for data. Utilization can drop below 20–30%.

This is exactly why single-request inference on a 70B model feels slow even on an H100 — the hierarchy is starved.

### 2\. What Batching Does: Increases Arithmetic Intensity

**Batching** = process B independent sequences (or requests) in parallel inside the **same kernel** (batched GEMM).

From first principles:

- The **weights** (A or B matrix in the GEMM) are **broadcast** to all B sequences.
- You load the weights from HBM **once** (or once per layer), then reuse them across the entire batch dimension.
- The compute work scales linearly with B (B × more matrix multiplies), but the memory traffic for weights scales only as ~1× (or sub-linearly with clever kernel fusion).

Mathematically (Roofline model perspective):

- Arithmetic intensity I = FLOPs / bytes moved from HBM.
- Without batching: I ≈ (model\_size × seq\_len) / (model\_size × tokens\_generated) → low I.
- With batching: I ≈ (B × seq\_len × hidden\_dim²) / model\_size → I scales ≈ linearly with B.

Higher I means you move fewer bytes from expensive HBM per FLOP performed → you saturate the tensor cores.

This directly exploits the hierarchy:

- Weights are fetched from HBM **once** and then:
	- Streamed through L2 (now serving many SMs efficiently).
		- Tiled into L1 / shared memory once per tile.
		- Reused thousands of times inside registers and tensor cores across the batch.

### 3\. How Batching Interacts with Each Level of the Hierarchy

- **Registers (per-thread)**: Batching increases the number of active warps/threads. Each thread still works on its own slice of the batch, keeping accumulators and partial sums in registers. More batch = better hiding of any residual latency inside the SM.
- **L1 / Shared Memory (per-SM)**: This is where the magic of **FlashAttention** and **kernel tiling** shines.
	- In a batched attention kernel, you load a tile of Q (from current token) + K/V tiles (from cache) into shared memory **once**.
		- The **same** K/V tile is then reused across **all B heads** and **all B sequences** inside the SM.
		- Larger batch allows bigger effective tiles or more parallel warps without increasing HBM traffic proportionally.
		- Result: massive reduction in HBM round-trips per token generated. Shared memory bandwidth (TB/s) is now fully utilized.
- **L2 Cache (chip-wide)**: With batching, L2 becomes far more effective as a “weight cache” or intermediate buffer.
	- Weights that miss in L1 are more likely to hit in L2 because the same weights are being requested by **many SMs simultaneously** (coalesced, high reuse).
		- Activations and KV cache accesses become more cache-friendly (structured batch dimension).
- **HBM (global memory)**: This is the level that benefits most.
	- Weight loading from HBM is amortized over B sequences.
		- KV cache traffic still grows with B × seq\_len, but **continuous batching** (vLLM, TGI, etc.) and **PagedAttention** keep it manageable by only keeping “live” pages in HBM and swapping as needed.
		- Net effect: HBM bandwidth utilization goes from “wasted on low-intensity single-sequence traffic” to “saturating the 2+ TB/s roofline”.

### 4\. Why This Matters So Much in Practice (Numbers + Real-World)

- On an H100 (HBM3e ~3.35 TB/s, ~1000+ TFLOPs FP16):
	- Batch=1: you might achieve only 100–200 tokens/sec for a 70B model (heavily HBM-bound).
		- Batch=32–256 (typical inference): 1000–5000+ tokens/sec because you now saturate tensor cores.
- Techniques like **FlashAttention-2/3**, **TMA** (Tensor Memory Accelerator), and **quantization** (4-bit/8-bit weights) are all built on top of batching — they only work well when arithmetic intensity is already boosted by batching.
- Without batching, even perfect L1 tiling can’t overcome the fundamental HBM tax. With batching, the hierarchy “feels” like the compute is closer because reuse happens at every level.

**Continuous / dynamic batching** (the modern standard) further optimizes this by:

- Keeping the GPU’s SMs 100% occupied (no idle time waiting for new prompts).
- Dynamically grouping requests of similar lengths → better coalescing into L1/Shared.

### Summary Table: Batching’s Effect on the Hierarchy

| Hierarchy Level | Batch=1 Effect | Batched (B>1) Effect | Why It Improves |
| --- | --- | --- | --- |
| Registers | Under-utilized (few warps) | Fully saturated; more parallelism | Hides latency |
| L1 / Shared | Small tiles, low reuse per HBM load | Larger effective reuse across batch | Tiles stay hot |
| L2 | Poor hit rate (single-sequence traffic) | High hit rate (weights shared across SMs) | Amortizes misses |
| HBM | Dominates time (low arithmetic intensity) | Amortized; bandwidth now feeds real compute | Main win |

**Bottom line from first principles**: The GPU memory hierarchy exists because physics forces a speed-density trade-off. LLM inference is fundamentally memory-movement dominated. Batching is the cheapest, most effective way to increase **reuse at every level** of that hierarchy — turning one expensive HBM load into thousands of cheap register/L1 operations. That is why every production inference engine (vLLM, Hugging Face TGI, TensorRT-LLM, etc.) obsesses over batch size, continuous batching, and paged KV cache. Without it, you are leaving the entire hierarchy starved.

define arithmetic intensity from first principles using a concrete example.

**Arithmetic intensity (AI)**, also called operational intensity, is a fundamental metric in computer architecture and performance modeling. It quantifies **how much computation you get for every byte of data moved from (or to) memory**.

### Definition from First Principles

At the lowest level, every program does two things:

- **Computation**: Floating-point operations (FLOPs) such as additions, multiplications, multiply-accumulates (MACs), etc. Each MAC in neural networks typically counts as **2 FLOPs** (one multiply + one add).
- **Data movement**: Bytes transferred across a memory boundary — usually from the slowest relevant level in the hierarchy (e.g., HBM/global memory on a GPU, since that’s the expensive part).

**Arithmetic Intensity = Total FLOPs performed ÷ Total Bytes transferred from memory**

Unit: **FLOPs per byte** (FLOP/B).

This ratio tells you the **computational efficiency** of an algorithm relative to its memory traffic. It directly connects to the **Roofline model**, which plots attainable performance (FLOPs/s on y-axis) against arithmetic intensity (x-axis). The model has two limits:

- A horizontal “compute roof” = peak FLOPs/s of the hardware (tensor cores, etc.).
- A sloped “memory roof” = peak memory bandwidth (bytes/s) × arithmetic intensity.

If your AI is low, you hit the memory roof first (memory-bound: GPU spends time waiting on HBM). If high, you hit the compute roof (compute-bound: tensor cores are fully utilized).

The key insight from physics: moving a byte from HBM to the SM costs far more energy and time than performing a FLOP inside registers or L1. High AI means you amortize that expensive movement over lots of cheap local computation.

### Concrete Example: General Matrix Multiplication (GEMM)

GEMM (C = A × B) is the heart of LLM layers (projections, FFNs, attention scores). It perfectly illustrates how AI scales with problem size and why batching helps.

Consider two matrices:

- A: M × K
- B: K × N
- C: M × N (output)

**Computation (FLOPs)**:

- For each of the M×N output elements, you do K multiplications and (K-1) additions → roughly **2 × M × N × K FLOPs** (standard counting).

**Memory traffic (Bytes)** — assuming no reuse/caching for the naive case, and FP16 (2 bytes per element):

- Read A: M × K elements → 2 × M × K bytes
- Read B: K × N elements → 2 × K × N bytes
- Write C: M × N elements → 2 × M × N bytes
- Total bytes ≈ 2 × (M×K + K×N + M×N)

**Arithmetic Intensity**:

$$
AI = \frac{2 \cdot M \cdot N \cdot K}{2 \cdot (M \cdot K + K \cdot N + M \cdot N)} = \frac{M \cdot N \cdot K}{M \cdot K + K \cdot N + M \cdot N}
$$

#### Small example (low AI, memory-bound)

Let M = N = K = 128 (tiny matrices, like a small vector-matrix multiply or early-layer case):

$$
AI = \frac{128 \cdot 128 \cdot 128}{128\cdot128 + 128\cdot128 + 128\cdot128} = \frac{2{,}097{,}152}{49{,}152} \approx 42.7 \text{ FLOP/B}
$$

(Actually lower in practice due to overheads; for even smaller or matrix-vector, it drops below 10 or even <1.)

On a modern GPU like H100 (peak ~2,000+ TFLOPs FP16, HBM bandwidth ~3+ TB/s), the “ridge point” (where compute and memory roofs meet) is around 500–1000+ FLOP/B depending on precision. Low AI like this → heavily memory-bound. Most time is spent loading data from HBM rather than computing.

#### Large example (high AI, compute-bound)

Let M = N = K = 8192 (large square GEMM, common in batched LLM layers):

$$
AI \approx \frac{8192^3}{3 \times 8192^2} \approx \frac{8192}{3} \approx 2730 \text{ FLOP/B}
$$

Now AI is very high → the kernel becomes compute-bound. You perform thousands of FLOPs per byte loaded. Tensor cores stay busy; HBM bandwidth is no longer the limiter.

Notice: **AI scales with matrix size** (roughly O(N) for square matrices). Larger problems = better reuse of data once loaded into faster levels (L2 → L1/shared → registers).

### Connection to LLM Inference and Batching (from our prior discussion)

In LLM decoding (autoregressive generation):

- Model weights are huge and live in HBM.
- For **batch size = 1**: Each token does roughly 2 × #parameters FLOPs, but you reload (or touch) almost the entire weight matrix from HBM. AI stays relatively low (~50–100 FLOP/B for typical Llama-scale models in decode phase).
- With **batching (batch size B)**: Weights are loaded once from HBM and reused across all B sequences. Compute scales with B, but weight memory traffic stays ~constant. → AI increases roughly proportionally to B (until other limits like KV cache or SM occupancy kick in).

This is why continuous batching and larger effective batches dramatically improve tokens/second: they push the workload rightward on the Roofline diagram, moving from the sloped memory roof toward the flat compute roof.

**FlashAttention** further boosts effective AI by tiling Q/K/V into shared memory (L1) and doing the entire attention computation with minimal HBM round-trips — again, maximizing FLOPs per byte moved from the slow hierarchy level.

### Why This Metric Matters So Much

- **Predicts bottlenecks**: Low AI (< ridge point) → optimize for less memory traffic (tiling, fusion, quantization, better caching). High AI → focus on compute (better kernels, more parallelism).
- **Guides optimizations**: Kernel fusion, operator reordering, quantization (fewer bytes per value), and batching all exist to increase AI.
- **Hardware-algorithm co-design**: GPUs excel at high-AI workloads like dense GEMMs because of massive tensor cores and wide HBM interfaces. Low-AI ops (e.g., elementwise, reductions) are painful.

In short: **Arithmetic intensity is the “FLOPs-per-byte efficiency”** of your algorithm. It directly reveals whether your code (or LLM layer) is starving the compute units due to excessive data movement through the memory hierarchy — or whether it’s efficiently reusing data once it’s close to the tensor cores in registers and L1.

For most LLM inference kernels, the goal is simple: **maximize reuse at every level** (HBM → L2 → L1/Shared → Registers) so AI becomes as high as possible. That’s the entire reason batching, FlashAttention, and paged KV cache exist.