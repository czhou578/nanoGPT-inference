# First-Principles Multimodal Inference Roadmap

The goal is to implement advanced multimodal inference concepts (from `README.md`) in pure, understandable Python. 

## Answering your question: Isolated vs Integrated?

**TL;DR:** Do not build them completely separately. Build an iterative, evolving system.

If you build them completely isolated in their own folders, you will struggle to combine them later because the interfaces of these systems are highly coupled. For example, **Continuous Batching** is intrinsically linked with how the **Scheduler** works, and the Scheduler needs the **PagedAttention / KV Cache Manager** to track memory blocks.

**The most valuable approach** is the "Onion Strategy". You build a functional, *naive* inference loop first. Then, you iteratively swap out naive implementations with your advanced, first-principles implementations. 

---

## Proposed Implementation Plan

Below is the structured roadmap to build these concepts from scratch seamlessly.

### Phase 1: The Naive Foundation
Build a basic building block that acts as our skeleton.
- **Components to build:** 
  - Dummy model weights or a basic TinyLlama/CLIP setup via PyTorch.
  - A naive `KVCache` that just appends tensors.
  - A simple loop that processes one request at a time (Prefill then Decode).
- **Outcomes:** A working visualizer backend that takes text + image and spits out text slowly. 

### Phase 2: Memory & Batching (The vLLM Core)
Replace the naive parts with high-efficiency structures.
- **[NEW] `paged_attention.py`**: Implement a Block Manager that creates a pool of memory blocks (e.g., 16 tokens per block). Update your attention mechanism to fetch KV pairs from these scattered blocks instead of contiguous arrays.
- **[NEW] `scheduler.py`**: A system that decides which requests get to run based on available blocks.
- **[NEW] `continuous_batching.py`**: Modify the main loop to be event-driven. Instead of waiting for a batch to finish, it pauses a sequence when it emits an EOS token, and the Scheduler immediately slots in a waiting request in the next token-generation step.

### Phase 3: Optimizing the Pipeline (Chunking & Streaming)
Now that batching and memory are robust, optimize the runtime constraints.
- **[MODIFY] `scheduler.py` (Chunked Prefill & Decode)**: Instead of processing a 2000-token prompt in one massive prefill iteration (which starves decodes in other requests), allow the scheduler to process prefill prompts in 512-token chunks, mixing them in the same forward pass as 1-token decodes from other sequences.
- **[NEW] `context_streaming.py`**: Implement a Ring-Buffer mechanism for the KV cache. When a sequence hits the maximum sequence length, start evicting the oldest blocks (while keeping the "attention sink" like the first 4 tokens).

### Phase 4: Disaggregation & Scale
Introduce architectural separation.
- **[NEW] `vision_encoder.py` (Encoder Disaggregation & Caching)**: Extract the vision model into a completely separate Python Process or thread. Expose an queue-based interface. Add a **Vision Encoding Cache** by hashing incoming images to bypass the encoder if seen previously.
- **[NEW] `parallelism.py`**: Simulate multiple devices. You can use Python `multiprocessing` arrays to simulate Tensor Parallelism (splitting matrix multiplications across two processes) or Pipeline Parallelism (Process 1 computes layers 1-4, then sends intermediate acts to Process 2 for layers 5-8).

## Open Questions

- **Framework**: We can use PyTorch tensors for array manipulations to keep the math easy without relying on external compiled C++ kernels. Does using PyTorch core functions (like `torch.matmul` and `torch.nn.functional.softmax`) align with your definition of "first principles", or do you want to write raw matrix multiplications in bare NumPy/Python?
- **Weights**: Would you like to use dummy random weights for this visualizer to strictly focus on system architecture, or load real mini-weights (like tiny-llama and a tiny-clip) so that the UI generates coherent outputs?

## Verification Plan

### Automated Tests
- Writing unit tests for the `BlockManager` to ensure memory allocations and frees happen correctly and sequence tables point to the right logical chunks.
- Validating the mathematical outputs of the `PagedAttention` function against a standard array approach.

### Manual Verification
- Using the React Frontend to submit heavy loads (e.g. 10 rapid-fire requests) and visually confirming in the UI that the system interleaves token generation via Continuous Batching.
