"""
Baseline vs KV-cache benchmark for your current nanoGPT file.

This version matches the implementation where:
- model.forward(...) returns only:
      logits, loss
- KV cache lives inside each Head as:
      head.key_cache
      head.value_cache
- cached generation is enabled by model.eval()
- non-cached generation is forced by model.train()
- model.forward accepts:
      start_pos=...

Put this file next to your nanoGPT script or in your benchmarks/ folder.
"""

from dataclasses import dataclass
import time
import torch
import torch.nn.functional as F


@dataclass
class GenerationTiming:
    name: str
    total_tokens: int
    total_seconds: float
    time_to_first_token: float

    @property
    def tokens_per_second(self):
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_tokens / self.total_seconds


def _sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_next_token(logits, generator=None):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _clear_kv_cache(model):
    """
    Works with your internal-cache Head implementation.
    """
    for module in model.modules():
        if hasattr(module, "key_cache"):
            module.key_cache = None
        if hasattr(module, "value_cache"):
            module.value_cache = None


@torch.no_grad()
def generate_no_cache_instrumented(
    model,
    idx,
    max_new_tokens,
    *,
    block_size,
    device,
    generator=None,
):
    """
    No-cache baseline.

    Forces model.train() so your Head.forward() uses the normal causal-mask path
    instead of the internal KV-cache path.
    """
    was_training = model.training
    model.train()
    _clear_kv_cache(model)

    first_token_seconds = None

    _sync_if_cuda(device)
    start = time.perf_counter()

    for token_i in range(max_new_tokens):
        step_start = time.perf_counter()

        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond, start_pos=0)
        logits = logits[:, -1, :]
        idx_next = _sample_next_token(logits, generator=generator)
        idx = torch.cat((idx, idx_next), dim=1)

        _sync_if_cuda(device)

        if token_i == 0:
            first_token_seconds = time.perf_counter() - step_start

    if not was_training:
        model.eval()

    _sync_if_cuda(device)
    total_seconds = time.perf_counter() - start

    return idx, GenerationTiming(
        name="no_cache",
        total_tokens=max_new_tokens,
        total_seconds=total_seconds,
        time_to_first_token=first_token_seconds or 0.0,
    )


@torch.no_grad()
def generate_with_internal_kv_cache_instrumented(
    model,
    idx,
    max_new_tokens,
    *,
    device,
    generator=None,
):
    """
    Cached generation for your internal Head cache.

    Important invariant:
    - prefill runs the full prompt once
    - each decode step feeds only the token sampled on the previous step
    - `start_pos` is the absolute position of that one-token input
    """
    was_training = model.training
    model.eval()
    _clear_kv_cache(model)

    B, prompt_len = idx.shape
    assert B == 1, "This first benchmark is single-request only."
    assert prompt_len > 0, "Prompt must contain at least one token."

    first_token_seconds = None

    _sync_if_cuda(device)
    start = time.perf_counter()

    # Prefill prompt once.
    first_start = time.perf_counter()
    logits, _ = model(idx, start_pos=0)
    logits = logits[:, -1, :]
    idx_next = _sample_next_token(logits, generator=generator)
    idx = torch.cat((idx, idx_next), dim=1)

    _sync_if_cuda(device)
    first_token_seconds = time.perf_counter() - first_start

    # Decode remaining tokens. We already emitted token 1 from prefill logits.
    for _ in range(1, max_new_tokens):
        # The new token we are feeding is at position len(idx) - 1.
        start_pos = idx.shape[1] - 1

        logits, _ = model(idx[:, -1:], start_pos=start_pos)
        logits = logits[:, -1, :]
        idx_next = _sample_next_token(logits, generator=generator)
        idx = torch.cat((idx, idx_next), dim=1)

    if was_training:
        model.train()

    _sync_if_cuda(device)
    total_seconds = time.perf_counter() - start

    return idx, GenerationTiming(
        name="kv_cache",
        total_tokens=max_new_tokens,
        total_seconds=total_seconds,
        time_to_first_token=first_token_seconds,
    )


def print_benchmark_table(rows):
    headers = ["method", "tokens", "wall_time_s", "tokens_per_s", "ttft_ms"]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.tokens_per_second:.2f}",
            f"{row.time_to_first_token * 1000:.2f}",
        ])

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rendered))
        for i in range(len(headers))
    ]

    def fmt(values):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rendered:
        print(fmt(row))

    if len(rows) == 2:
        baseline, cached = rows
        speedup = cached.tokens_per_second / baseline.tokens_per_second
        print()
        print(f"KV-cache throughput speedup: {speedup:.2f}x")


def run_baseline_vs_kv_benchmark(
    model,
    prompt_tokens,
    *,
    N=64,
    device,
    block_size,
    seed=1337,
    temperature=1.0,
):
    """
    Compatible replacement for your current benchmark entry point.

    `temperature` is accepted for API compatibility but intentionally unused.
    """
    del temperature

    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Keep these runs reproducible.
    gen_no_cache = torch.Generator(device=device)
    gen_no_cache.manual_seed(seed)

    gen_kv = torch.Generator(device=device)
    gen_kv.manual_seed(seed)

    _, no_cache_timing = generate_no_cache_instrumented(
        model,
        prompt.clone(),
        N,
        block_size=block_size,
        device=device,
        generator=gen_no_cache,
    )

    _, kv_timing = generate_with_internal_kv_cache_instrumented(
        model,
        prompt.clone(),
        N,
        device=device,
        generator=gen_kv,
    )

    print_benchmark_table([no_cache_timing, kv_timing])

    return {
        "no_cache": no_cache_timing,
        "kv_cache": kv_timing,
    }
