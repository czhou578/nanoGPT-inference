# 🧠 Multimodal Inference Visualizer

The goal for me is to understand from first principles how multimodal inference works, using the example of the vLLM project. This is supposed to be as simple as possible, showing every step from its basic building blocks / first principles.  

The user will be able to upload an image and some text and see how the input is processed and how the output is generated. 

## Tech Stack

- FastAPI
- Uvicorn
- React
- Vite

## Concepts 

The following concepts are explored:

- **Continuous batching**: Instead of waiting for all requests in a static batch to finish generation, continuous batching dynamically adds new requests to the batch at the token level as soon as other sequences complete. This dramatically increases GPU utilization and throughput by minimizing idle compute.
- **PagedAttention**: An attention algorithm inspired by operating system virtual memory. It partitions the Key-Value (KV) cache into non-contiguous, fixed-size blocks (pages). This eliminates memory fragmentation and avoids pre-allocating memory for maximum sequence lengths, allowing the system to batch many more requests simultaneously.
- **KV cache**: In autoregressive models, generating a new token depends on all previous tokens. The KV cache stores the previously computed Key and Value tensors for past tokens so they don't have to be recomputed at each step, significantly speeding up the decoding phase.
- **Scheduling**: The intelligent allocation of GPU resources and memory. A sequence scheduler dictates which requests to execute, preempts (pauses) requests when memory limits are reached, swaps memory to CPU, and resumes generation, ensuring maximum efficiency without out-of-memory errors.
- **Chunked Prefill and Decode stages**: LLM inference has two phases: Prefill (processing the input prompt, which is compute-heavy) and Decode (generating output tokens one-by-one, which is memory-bandwidth bound). *Chunked Prefill* splits long input prompts into smaller blocks, allowing prefill computations to be interleaved alongside memory-bound decode computations to maximize hardware efficiency.
- **Encoder disaggregation**: In multimodal pipelines, vision encoders (like CLIP) extract features from images, while LLMs process text. Disaggregation decouples these two components, allowing them to scale independently or run on specialized hardware (e.g., executing the vision encoder separately from the massive language model weights).
- **Parallelism**: Distributing model inference across multiple GPUs to handle massive memory and compute requirements. This includes *Tensor Parallelism* (splitting individual network layers across GPUs) and *Pipeline Parallelism* (assigning sequential layers to different GPUs).
- **Vision encoding cache**: Caching the high-dimensional feature representations produced by the vision encoder. If the same image is re-used (e.g., in a multi-turn chat involving a visual context or system prompt), caching prevents redundant and expensive visual feature extraction.
- **Context streaming**: Techniques for managing extremely long inputs by streaming context or selectively discarding older KV cache blocks. This keeps memory usage bounded, enabling models to maintain conversational state or process long documents with a finite hardware footprint.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

