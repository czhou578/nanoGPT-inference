"""
CUDA Graph vs Eager decode benchmark.

Compares three generation strategies:
  1. No cache     — full-context recompute every step (model.train() path)
  2. Eager KV     — KV cache with eager decode (generate_kv_cache)
  3. CUDA Graph   — KV cache with graph-captured decode (generate_cuda_graph)

Uses torch.cuda.Event for precise GPU timing (not wall-clock).
"""

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    name: str
    total_tokens: int
    total_ms: float
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    warmup_capture_ms: float = 0.0
    per_step_ms: list = field(default_factory=list)

    @property
    def tokens_per_second(self):
        if self.total_ms <= 0:
            return float("inf")
        return self.total_tokens / (self.total_ms / 1000.0)

    @property
    def avg_decode_step_ms(self):
        if not self.per_step_ms:
            return 0.0
        return sum(self.per_step_ms) / len(self.per_step_ms)

    @property
    def median_decode_step_ms(self):
        if not self.per_step_ms:
            return 0.0
        s = sorted(self.per_step_ms)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2


def _gpu_timer():
    """Return a (start, end) pair of CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


# ──────────────────────────────────────────────────────────
# Benchmark 1: Eager KV-cache generation (per-step timing)
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def bench_eager_kv(model, idx, max_new_tokens, *, clear_cache_fn, block_size):
    """
    Benchmark eager KV-cache decode with per-step GPU timing.
    """
    model.eval()
    clear_cache_fn(model)

    T_prompt = idx.shape[1]
    per_step_ms = []

    # ── Prefill ──
    prefill_start, prefill_end = _gpu_timer()
    prefill_start.record()
    logits, _ = model(idx, cache_pos=0)
    prefill_end.record()
    torch.cuda.synchronize()
    prefill_ms = prefill_start.elapsed_time(prefill_end)

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
    cache_pos = T_prompt

    # ── Decode ──
    total_start, total_end = _gpu_timer()
    total_start.record()

    for _ in range(max_new_tokens - 1):
        step_start, step_end = _gpu_timer()
        step_start.record()

        logits, _ = model(idx_next, start_pos=cache_pos, cache_pos=cache_pos)

        step_end.record()
        torch.cuda.synchronize()
        per_step_ms.append(step_start.elapsed_time(step_end))

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        cache_pos += 1

    total_end.record()
    torch.cuda.synchronize()
    decode_ms = total_start.elapsed_time(total_end)

    model.train()
    return idx, BenchmarkResult(
        name="eager_kv_cache",
        total_tokens=max_new_tokens,
        total_ms=prefill_ms + decode_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        per_step_ms=per_step_ms,
    )


# ──────────────────────────────────────────────────────────
# Benchmark 2: CUDA graph generation (per-step timing)
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def bench_cuda_graph(model, idx, max_new_tokens, *, clear_cache_fn):
    """
    Benchmark CUDA graph decode with per-step GPU timing.
    Includes warmup + capture cost measured separately.
    """
    model.eval()
    clear_cache_fn(model)

    T_prompt = idx.shape[1]
    per_step_ms = []

    # ── Prefill (eager) ──
    prefill_start, prefill_end = _gpu_timer()
    prefill_start.record()
    logits, _ = model(idx, cache_pos=0)
    prefill_end.record()
    torch.cuda.synchronize()
    prefill_ms = prefill_start.elapsed_time(prefill_end)

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
    cache_pos = T_prompt

    # ── Warmup + Capture ──
    warmup_start, warmup_end = _gpu_timer()
    warmup_start.record()

    model.static_input_ids.copy_(idx_next)
    model.static_position.fill_(cache_pos)
    model.static_cache_pos.fill_(cache_pos)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        static_output = model.decode_one_token()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        static_output = model.decode_one_token()

    warmup_end.record()
    torch.cuda.synchronize()
    warmup_capture_ms = warmup_start.elapsed_time(warmup_end)

    cache_pos += 1

    # ── Decode loop (graph replay) ──
    total_start, total_end = _gpu_timer()
    total_start.record()

    for _ in range(max_new_tokens - 1):
        model.static_input_ids.copy_(idx_next)
        model.static_position.fill_(cache_pos)
        model.static_cache_pos.fill_(cache_pos)

        step_start, step_end = _gpu_timer()
        step_start.record()

        graph.replay()

        step_end.record()
        torch.cuda.synchronize()
        per_step_ms.append(step_start.elapsed_time(step_end))

        logits = static_output[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        cache_pos += 1

    total_end.record()
    torch.cuda.synchronize()
    decode_ms = total_start.elapsed_time(total_end)

    model.train()
    return idx, BenchmarkResult(
        name="cuda_graph",
        total_tokens=max_new_tokens,
        total_ms=prefill_ms + decode_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        warmup_capture_ms=warmup_capture_ms,
        per_step_ms=per_step_ms,
    )


# ──────────────────────────────────────────────────────────
# Pretty-print
# ──────────────────────────────────────────────────────────

def print_benchmark_table(results):
    """Print a comparison table of benchmark results."""
    print()
    headers = [
        "method", "tokens", "total_ms", "tok/s",
        "prefill_ms", "decode_ms", "avg_step_ms", "median_step_ms",
        "warmup_ms",
    ]

    rows = []
    for r in results:
        rows.append([
            r.name,
            str(r.total_tokens),
            f"{r.total_ms:.2f}",
            f"{r.tokens_per_second:.1f}",
            f"{r.prefill_ms:.2f}",
            f"{r.decode_ms:.2f}",
            f"{r.avg_decode_step_ms:.3f}",
            f"{r.median_decode_step_ms:.3f}",
            f"{r.warmup_capture_ms:.2f}" if r.warmup_capture_ms > 0 else "—",
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(values):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    # Speedup summary
    if len(results) == 2:
        baseline, optimized = results
        if baseline.tokens_per_second > 0:
            speedup = optimized.tokens_per_second / baseline.tokens_per_second
            step_speedup = (
                baseline.avg_decode_step_ms / optimized.avg_decode_step_ms
                if optimized.avg_decode_step_ms > 0 else float("inf")
            )
            print()
            print(f"  End-to-end throughput speedup: {speedup:.2f}x")
            print(f"  Per-step decode speedup:       {step_speedup:.2f}x")


def run_eager_vs_cuda_graph_benchmark(
    model,
    prompt_tokens,
    *,
    N,
    clear_cache_fn,
    block_size,
    device,
    seed=1337,
):
    """
    Run one eager-vs-CUDA-graph comparison.

    Args:
        model: Trained GPTLanguageModel on CUDA.
        prompt_tokens: List[int] — the prompt token IDs.
        N: Number of tokens to generate.
        clear_cache_fn: Function that zeros the KV caches.
        block_size: Max context length.
        device: Device string.
        seed: Random seed for reproducibility.
    """
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, eager_result = bench_eager_kv(
        model, prompt.clone(), N,
        clear_cache_fn=clear_cache_fn,
        block_size=block_size,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _, graph_result = bench_cuda_graph(
        model, prompt.clone(), N,
        clear_cache_fn=clear_cache_fn,
    )

    print_benchmark_table([eager_result, graph_result])
    return {"eager_kv": eager_result, "cuda_graph": graph_result}
