# Pipeline Parallelism: From First Principles

Source: Reiner Pope podcast discussion (~00:47:12), Huang et al. *GPipe* (2019), Narayanan et al. *PipeDream* (2019)

---

## 1. The Problem Pipeline Parallelism Solves

A modern LLM is a stack of transformer layers. Llama 3 70B has 80 layers. In FP16, the weights alone occupy ~140 GB of HBM.

A single H100 has 80 GB of HBM. **The model literally does not fit on one device.**

You need to spread the model across multiple devices. There are three main strategies:

| Strategy | What gets split | Communication pattern |
|---|---|---|
| **Tensor parallelism (TP)** | Individual weight matrices (columns/rows) | All-reduce after every layer — requires ultra-fast interconnect (NVLink) |
| **Pipeline parallelism (PP)** | Contiguous groups of layers | Point-to-point activation transfer between stages — lower bandwidth requirement |
| **Data parallelism (DP)** | Nothing — full model replicated | Gradient all-reduce after backward pass |

Pipeline parallelism is the natural choice when you need to span **across nodes/racks** where interconnect bandwidth is lower (e.g., InfiniBand at 400 Gb/s vs. NVLink at 900 GB/s). You assign contiguous chunks of layers to each device (called a **stage**), and data flows through the stages sequentially, like an assembly line.

---

## 2. Intra-Node vs. Inter-Node Communication — From First Principles

Before understanding *why* pipeline parallelism is used (and where), you need to understand the physical topology that GPUs live in. The entire rationale for choosing PP over TP (or vice versa) comes down to one fact: **not all GPU-to-GPU links are created equal.**

### What is a "node"?

A **node** is a single physical server — one motherboard, one (or two) CPUs, and a fixed number of GPU accelerators sharing a local interconnect. Think of it as one box you could pick up and carry (if it weren't bolted into a rack).

A typical AI training node:
- **NVIDIA DGX H100**: 8× H100 GPUs in one node
- **NVIDIA DGX B200**: 8× B200 GPUs in one node

All 8 GPUs inside a node are connected to each other via a **dedicated, high-speed fabric** that lives entirely within the physical chassis.

### Intra-node communication: NVLink

The GPUs inside a single node communicate over **NVLink**, NVIDIA's proprietary point-to-point interconnect:

| Property | NVLink (H100 generation) |
|---|---|
| Bandwidth (bidirectional) | **900 GB/s** per GPU |
| Latency | ~1–5 μs |
| Topology | Full mesh (every GPU can talk to every other GPU directly) |
| Physical medium | High-speed traces on the baseboard / NVSwitch chips |

At 900 GB/s, an 8-GPU node has an **aggregate internal bisection bandwidth** of several TB/s. This is enormous — it means GPUs inside a node can exchange data almost as fast as they can read from their own HBM.

**Why this matters for parallelism**: Tensor parallelism (TP) requires an **all-reduce** operation after every single layer. For a 70B model with 80 layers, that's 80 all-reduces per forward pass. Each all-reduce exchanges partial activation results between all participating GPUs. This only works if the communication is nearly instantaneous — and at 900 GB/s with microsecond latency, NVLink makes it viable.

### Inter-node communication: InfiniBand / Ethernet

When you need more than 8 GPUs (which you almost always do for large models), you must connect **multiple nodes** together. These nodes communicate over **network fabric**:

| Property | InfiniBand (NDR 400G) | RoCE / Ethernet (400GbE) |
|---|---|---|
| Bandwidth per link | **50 GB/s** (400 Gb/s) | **50 GB/s** (400 Gb/s) |
| Effective bandwidth per GPU | ~25–50 GB/s (shared) | ~12–50 GB/s |
| Latency | ~1–5 μs (RDMA) | ~5–20 μs |
| Topology | Fat-tree or dragonfly via switches | Leaf-spine via switches |
| Physical medium | Cables between racks, through switches |

The critical numbers: a single GPU's inter-node bandwidth is roughly **25–50 GB/s**, compared to **900 GB/s** intra-node. That's a **~20–35× gap**.

### The bandwidth cliff visualized

```
  Bandwidth (GB/s, log scale)
  ─────────────────────────────────────
  3,350 │ ██████████████████████  HBM (local memory)
    900 │ ██████████████         NVLink (intra-node)
     50 │ ██                     InfiniBand (inter-node)
     12 │ █                      Ethernet (across data center)
```

Every time you cross a physical boundary — from on-chip to on-node to across-node — you lose roughly an order of magnitude of bandwidth. This is the **memory wall** extended to the network: the further away the data is, the slower it arrives.

### Why this bandwidth gap creates the need for pipeline parallelism

Recall the parallelism strategies from Section 1:

**Tensor parallelism** splits weight matrices and requires an all-reduce after every layer. The volume of data exchanged per all-reduce is proportional to the activation size (roughly $B \times d_{\text{model}} \times \text{bytes}$). For Llama 3 70B with hidden dimension 8,192 and batch size 32 in FP16:

$$
\text{All-reduce volume per layer} \approx 2 \times B \times d_{\text{model}} \times 2 \text{ bytes} = 2 \times 32 \times 8192 \times 2 \approx 1 \text{ MB}
$$

With 80 layers, that's ~80 MB per forward pass. On NVLink at 900 GB/s, this takes:

$$
t_{\text{comm}} = \frac{80 \times 10^6}{900 \times 10^9} \approx 0.09 \text{ ms} \quad \text{(negligible)}
$$

On InfiniBand at 50 GB/s:

$$
t_{\text{comm}} = \frac{80 \times 10^6}{50 \times 10^9} \approx 1.6 \text{ ms} \quad \text{(still small in absolute terms...)}
$$

But the real killer is **latency**: each all-reduce is a synchronous barrier. With 80 layers, you're hit with 80 round-trip latencies. At ~5 μs per NVLink all-reduce that's 0.4 ms total. At ~10-20 μs per InfiniBand all-reduce, that's 0.8–1.6 ms of pure latency overhead — and this stacks on top of the bandwidth cost. The communication starts to eat a meaningful fraction of compute time.

**Pipeline parallelism** sends activations only **once per stage boundary** — a single point-to-point transfer between adjacent stages. If you have 4 pipeline stages, that's only 3 inter-stage transfers per forward pass (vs. 80 all-reduces for TP). Each transfer is the same ~1 MB activation, taking ~20 μs over InfiniBand. Total: ~60 μs. This is why PP tolerates low-bandwidth links while TP cannot.

### The resulting design principle: TP inside, PP outside

This bandwidth asymmetry produces a universal design pattern in large-scale LLM systems:

```
┌─────────────────────────────────────────────────────┐
│                   Cluster / Pod                      │
│                                                      │
│  ┌──────────────┐    InfiniBand     ┌──────────────┐ │
│  │   Node 0     │◄────(PP)─────────►│   Node 1     │ │
│  │              │   ~50 GB/s        │              │ │
│  │  GPU0  GPU1  │                   │  GPU0  GPU1  │ │
│  │   ◄──TP──►   │                   │   ◄──TP──►   │ │
│  │  ~900 GB/s   │                   │  ~900 GB/s   │ │
│  │  GPU2  GPU3  │                   │  GPU2  GPU3  │ │
│  │   ◄──TP──►   │                   │   ◄──TP──►   │ │
│  │  GPU4  GPU5  │                   │  GPU4  GPU5  │ │
│  │   ◄──TP──►   │                   │   ◄──TP──►   │ │
│  │  GPU6  GPU7  │                   │  GPU6  GPU7  │ │
│  │   ◄──TP──►   │                   │   ◄──TP──►   │ │
│  └──────────────┘                   └──────────────┘ │
│                                                      │
│   Stage 0 (Layers 0–39)             Stage 1 (40–79)  │
└─────────────────────────────────────────────────────┘
```

- **Within each node**: 8 GPUs run tensor parallelism (TP=8). Each GPU holds a *shard* of every layer's weight matrices. All-reduces fly over NVLink at 900 GB/s. Latency is microseconds.
- **Across nodes**: Pipeline parallelism (PP). Node 0 computes layers 0–39, sends one activation tensor to Node 1, which computes layers 40–79. One point-to-point transfer over InfiniBand per stage transition.

This is the standard topology for serving models like Llama 3 70B (TP=8 within one node, or TP=8 + PP=2 across two nodes) and for training models like Llama 3 405B (TP=8, PP=4+ across many nodes, plus data parallelism across replicas).

### The key takeaway

Pipeline parallelism is not chosen because it is *better* than tensor parallelism. It is chosen because the **physical reality of interconnect bandwidth** makes TP infeasible across node boundaries. PP exists to work around the bandwidth cliff between intra-node and inter-node communication. Every design decision in this document — the bubble overhead, the choice of how many stages to use, the latency vs. throughput tradeoff — traces back to this single hardware constraint.

---

## 3. How Pipeline Parallelism Works

### The basic setup

Suppose we have a model with $L$ layers and $P$ pipeline stages (devices). Each stage holds $L/P$ consecutive layers.

For a 4-stage pipeline across 4 racks:

| Stage (Rack) | Layers held |
|---|---|
| Rack 0 | Layers 0–19 |
| Rack 1 | Layers 20–39 |
| Rack 2 | Layers 40–59 |
| Rack 3 | Layers 60–79 |

### The forward pass flow

When a batch of tokens enters the model:

1. **Rack 0** computes layers 0–19, producing an intermediate activation tensor.
2. Rack 0 **sends** that activation to **Rack 1** over the network.
3. **Rack 1** computes layers 20–39 on the received activation, sends it to Rack 2.
4. This continues until **Rack 3** produces the final output (logits).

The backward pass flows in reverse: gradients propagate from Rack 3 back to Rack 0.

### What each stage actually does per micro-batch

Each stage performs both a forward pass (F) and a backward pass (B) on each micro-batch. In the diagram from the podcast:

- **F0, F1, F2, F3** = forward passes on micro-batches 0, 1, 2, 3
- **B0, B1, B2, B3** = backward passes on micro-batches 0, 1, 2, 3

The key constraint: a stage cannot begin its forward pass on micro-batch $i$ until the *preceding* stage has finished its forward pass on micro-batch $i$ and sent the activations. Similarly, backward passes propagate in reverse order.

---

## 4. The Bubble Problem — From First Principles

### Why bubbles are unavoidable

Look at Rack 3 (the last stage) at the very beginning of a batch. It cannot do anything until Racks 0, 1, and 2 have each completed their forward passes sequentially and forwarded activations up the chain. During this startup period, **Rack 3 is completely idle**. This is the "fill" phase.

Symmetrically, at the end of the batch, Rack 0 finishes its backward pass first and then sits idle while Racks 1, 2, and 3 complete theirs. This is the "drain" phase.

The idle time during fill and drain is called the **pipeline bubble**.

```
rack
  3       |           F0   F1   F2   F3 | B3   B2   B1   B0
  2       |      F0   F1   F2   F3 |      B3   B2   B1   B0
  1       | F0   F1   F2   F3 |           B3   B2   B1   B0
  0       F0   F1   F2   F3 | bubble(drain)  B3   B2   B1   B0
          ──────────────────────────────────────────────→ time

  Legend: F = Forward pass, B = Backward pass, empty space = idle (bubble)
```

### Quantifying the bubble

If each micro-batch takes time $t$ for one forward or backward pass through one stage, then:

- **Useful work time per stage** = $M \times (t_F + t_B)$ where $M$ is the number of micro-batches
- **Bubble time** = $(P - 1) \times (t_F + t_B)$ — the startup/drain overhead from $P$ stages

The **bubble fraction** (fraction of total time wasted) is:

$$
\text{Bubble fraction} = \frac{P - 1}{M + P - 1}
$$

To make the bubble negligible, you need $M \gg P$. For example, with $P = 4$ stages:

| Micro-batches ($M$) | Bubble fraction |
|---|---|
| 4 | 43% |
| 8 | 27% |
| 16 | 16% |
| 64 | 4.5% |

This means you need to split each batch into many micro-batches to keep the pipeline efficient. But more micro-batches means smaller micro-batches, which can reduce per-device compute efficiency (less parallelism within each micro-batch).

---

## 5. Why Can't You Overlap Batches to Eliminate Bubbles?

During **training**, once all micro-batches in a batch have completed their forward and backward passes, you must:

1. **Aggregate gradients** across all micro-batches
2. **Update the model weights** (optimizer step)
3. Only *then* can you start the next batch

This is the **gradient synchronization barrier**. The weights used for micro-batch $i$ of batch $k+1$ must reflect the optimizer update from batch $k$. You cannot overlap batch $k+1$'s forward passes with batch $k$'s backward passes because batch $k+1$ would be using stale weights.

This is fundamentally different from **inference**, where there are no gradients and no weight updates. During inference, you *can* overlap work across different requests freely — which is exactly what continuous batching does.

> **Key distinction**: The bubble problem is primarily a **training** concern. During inference decode, each stage processes the same token in a strict sequential pipeline, and the "bubble" manifests differently (as pipeline latency rather than utilization loss).

---

## 6. The KV Cache Asymmetry — Why PP Splits Weights but Not KV Cache

This is a subtle but critical point that has major implications for system design.

### What pipeline parallelism divides

With $P$ pipeline stages, each stage holds $1/P$ of the model's layers. So the **weight memory** per device is reduced by a factor of $P$:

$$
\text{Weights per device} = \frac{N_{\text{total}} \times \text{bytes per param}}{P}
$$

For Llama 3 70B in FP16 across 4 stages: $140\text{ GB} / 4 = 35\text{ GB per device}$. Now it fits comfortably on an 80 GB H100.

### What pipeline parallelism does NOT divide

The **KV cache** is a different story. To keep a $P$-stage pipeline busy (avoid bubbles), you need at least $P$ micro-batches in flight simultaneously. Each micro-batch contains sequences that need their own KV cache entries.

Here's the reasoning:

1. While Rack 0 is computing the forward pass for micro-batch 3, Rack 1 is working on micro-batch 2, Rack 2 on micro-batch 1, and Rack 3 on micro-batch 0.
2. All $P$ micro-batches are in flight at the same time.
3. Each micro-batch's sequences need KV cache entries stored at *every* stage (since every stage computes its portion of attention for those sequences).

So the number of concurrent sequences scales with $P$, and every stage must hold KV cache for all concurrent sequences that pass through its layers. The net result:

$$
\text{KV cache per device} \propto \frac{L_{\text{stage}}}{L} \times P \times B_{\text{micro}} \times S \times d_{\text{kv}}
$$

The $L_{\text{stage}}/L$ factor (each stage has fewer layers → fewer KV heads to cache) is exactly cancelled by the $P$ factor (you need $P$ times as many concurrent sequences). **The per-device KV cache does not shrink with more pipeline stages.**

### Why this matters at long context lengths

For a concrete example with Llama 3 70B at 128K context:

- **Weights per device (4-way PP)**: 35 GB ✓ Fits easily
- **KV cache per sequence at 128K tokens**: ~40 GB (80 layers × 8 KV heads × 128 head_dim × 2 bytes × 128K tokens × 2 for K and V)
- With $P = 4$ micro-batches in flight, each device needs KV cache for its share of layers across all micro-batches

At long contexts, the KV cache dominates total memory. Pipeline parallelism helps with weights but **does not help with KV cache** — the dominant memory consumer. This severely limits its value for long-context workloads.

---

## 7. Pipeline Parallelism During Inference: A Different Problem

During training, pipeline parallelism is about overlapping forward and backward passes across micro-batches. During **inference decode**, the dynamics change:

### No backward pass
There are no gradients, so the pipeline is forward-only. This eliminates half the bubble.

### The latency penalty
For a single request, pipeline parallelism adds **latency**. One token generation requires the activation to pass through all $P$ stages sequentially. If each stage takes time $t$, the per-token latency is $P \times t$ (plus inter-stage communication). Compare to tensor parallelism, where all devices compute in parallel and the per-token latency is closer to $t$ (plus all-reduce communication).

### Where PP is used in inference
Pipeline parallelism is used in inference primarily when:
1. The model **doesn't fit on devices connected by fast interconnect** (e.g., must span across nodes)
2. You want to **minimize inter-node communication volume** — PP sends activations (small) rather than TP's all-reduce of partial results (also small, but more frequent)
3. You're willing to trade **higher latency for simpler communication patterns**

In practice, most large-scale inference systems use **TP within a node** (where NVLink provides ~900 GB/s) and **PP across nodes** (where InfiniBand provides ~50-100 GB/s effective bandwidth).

---

## 8. Investor Perspective: What to Pay Attention To

### The bubble tax on training efficiency

Pipeline bubbles represent **wasted GPU-hours that you're still paying for**. At scale, even a 5% bubble fraction across thousands of GPUs is enormous:

- 10,000 H100s training for 3 months at ~$2/GPU-hour
- Total cost: ~$43M
- 5% bubble waste: **~$2.2M in idle GPU time**

Companies that minimize bubble overhead (through techniques like interleaved scheduling, async pipeline methods, or simply using fewer pipeline stages with larger TP) have a real cost advantage.

### The KV cache wall limits PP's value for long-context inference

As context lengths push to 128K, 1M, and beyond:
- KV cache memory dominates over weight memory
- Pipeline parallelism doesn't reduce KV cache memory
- This means **PP alone cannot solve the memory problem for long-context serving**

Watch for companies investing in:
- **KV cache compression** (quantization, eviction policies, MLA/multi-latent attention)
- **Disaggregated serving** (separate prefill and decode clusters)
- **Offloading strategies** (KV cache to CPU/SSD)

### Interconnect bandwidth determines the PP vs. TP boundary

The choice between TP and PP at each level of the system hierarchy is dictated by available bandwidth:

| Interconnect | Bandwidth | Preferred parallelism |
|---|---|---|
| NVLink (within node) | ~900 GB/s | Tensor parallelism |
| InfiniBand (across nodes) | ~50-100 GB/s | Pipeline parallelism |
| Ethernet (across racks) | ~25-50 GB/s | Pipeline parallelism or data parallelism |

**Key signal**: The ratio of intra-node to inter-node bandwidth determines how many pipeline stages you're forced to use, and therefore how much bubble overhead you pay. Investments in faster inter-node interconnects (like NVIDIA's NVLink domain expansion, or custom optical interconnects) directly reduce the need for PP and its associated inefficiency.

### The training vs. inference divergence

Pipeline parallelism is **much more important for training** (where models must span many nodes and bubbles are a real cost) than for **inference** (where TP within a node often suffices, and PP adds latency). As the industry shifts investment from training to inference, the relative importance of PP-related optimizations changes:

- **Training-focused companies**: Watch their pipeline scheduling efficiency and bubble fraction
- **Inference-focused companies**: Watch their TP scaling, KV cache management, and continuous batching sophistication
- **Hardware companies**: Watch whether new chips prioritize intra-node bandwidth (helps TP/inference) vs. inter-node bandwidth (helps PP/training)

---

## 9. Industry Adoption and Trends (as of April 2026)

### Who uses PP for training

| Organization | Approach |
|---|---|
| **DeepSeek** | Pioneered **DualPipe** — a bidirectional pipeline scheduling algorithm that feeds micro-batches from *both ends* of the pipeline simultaneously, significantly reducing the bubble fraction compared to classic 1F1B scheduling. Heavily optimized for their MoE architectures (V3/R1), where the all-to-all expert routing communication makes bubble-hiding even more critical. |
| **Meta (Llama)** | Trained Llama 3 405B on 24K-GPU clusters using **4D parallelism** (TP + PP + DP + Context Parallelism). Their PP stages are carefully mapped to the physical network topology — pipeline stage boundaries are placed at rack/node boundaries where InfiniBand links live, while TP runs inside nodes over NVLink. |
| **Google (Gemini / TPUs)** | Developed **GPipe**, the foundational PP paper. Current Gemini training on TPUv5p pods uses the Pathways system with massive synchronous data parallelism. PP plays a *lesser* role on TPUs because TPU pod interconnects have a flatter bandwidth hierarchy — less of a cliff between intra-node and inter-node — so TP can be pushed further before hitting a bandwidth wall. |
| **OpenAI** | Uses multi-dimensional parallelism for frontier model training. Specifics are not published, but their scale (~10K+ GPUs) necessitates PP across node boundaries. |

### Who uses PP for inference

PP is secondary for inference (most single-node serving uses TP only), but it appears in multi-node deployments:

| Framework | PP Usage |
|---|---|
| **vLLM** | Supports multi-node PP for models that exceed a single node's aggregate HBM (e.g., 405B across 2+ nodes) |
| **SGLang** | Integrates PP for multi-node, multi-GPU serving |
| **TensorRT-LLM** | NVIDIA's inference engine; supports PP for cross-node model sharding |
| **Hugging Face TGI** | Supports PP for large model deployment |
| **DeepSpeed-Inference** | PP support for serving via the DeepSpeed ecosystem |

### Notable trends

**1. PP is now one dimension of "4D parallelism"**

No major lab uses PP in isolation anymore. The standard recipe is:
- **TP** within a node (NVLink, ~900 GB/s)
- **PP** across nodes (InfiniBand, ~50 GB/s)
- **DP** across replicas (for throughput scaling)
- **Context/Sequence Parallelism (CP)** for long-context training (partitioning the sequence dimension across devices)

The parallelism strategy is co-designed with the physical network topology. Each dimension of parallelism is placed at the level of the hardware hierarchy where its communication pattern is cheapest.

**2. DeepSeek's DualPipe is the most significant recent innovation**

Classic PP scheduling (1F1B) feeds micro-batches from one end of the pipeline. DualPipe feeds from *both* ends simultaneously:

```
Classic 1F1B:
  Stage 3:  ____________ F0  F1  F2  F3  B3  B2  B1  B0
  Stage 0:  F0  F1  F2  F3  B3  B2  B1  B0 ____________

DualPipe (conceptual):
  Stage 3:  F0' F1' F2' F3'  ...  B3  B2  B1  B0
  Stage 0:  F0  F1  F2  F3   ...  B3' B2' B1' B0'
  (both ends start immediately → smaller bubble)
```

This is especially important for MoE models where the expert all-to-all communication can be overlapped with computation during what would otherwise be bubble time.

**3. Google's TPU architecture partially obviates PP**

TPU pods connect chips via a custom **Inter-Chip Interconnect (ICI)** that provides relatively uniform bandwidth across 2D/3D torus topologies. Unlike GPU clusters where there's a 20×+ bandwidth cliff between NVLink and InfiniBand, TPU ICI is more homogeneous. This means Google can extend TP further across more chips before needing PP, reducing bubble overhead. However, at the largest scales (multi-pod training), PP is still necessary.

**4. Speculative decoding is being pipelined**

Recent work (2025–2026) pipelines the draft and verification steps in speculative decoding — the draft model runs on early pipeline stages while the verification model runs on later stages, overlapping their execution. This is an inference-specific use of PP that differs from the classical training formulation.

**5. Multimodal "modality bubbles" are the new frontier**

As models incorporate vision encoders, audio encoders, and LLM backbones, the different modality components have vastly different compute profiles. A vision encoder might take 10× longer than an LLM layer. Scheduling these heterogeneous components across a pipeline introduces "modality bubbles" — a new variant of the classic bubble problem that the community is actively working to solve.

---

## 10. Summary

| Concept | First-principle insight |
|---|---|
| **Pipeline parallelism** | Split model layers across devices; activations flow between stages |
| **Bubble** | Stages at the end of the pipeline are idle during fill; stages at the start are idle during drain |
| **Bubble fraction** | $(P-1)/(M+P-1)$ — need many micro-batches to amortize |
| **Can't overlap batches (training)** | Gradient synchronization barrier requires weight update before next batch |
| **KV cache not reduced by PP** | Need $P$ concurrent micro-batches → $P\times$ more sequences → KV cache scales back up |
| **Long context = PP is limited** | When KV cache >> weights in memory, splitting weights across stages doesn't help much |
| **Investor signal** | Watch interconnect bandwidth ratios, bubble overhead at scale, and KV cache management innovations |
