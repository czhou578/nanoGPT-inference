"""
Baseline vs KV-cache generation benchmark for a nanoGPT-style model.

Assumptions:
- Your model has the nanoGPT-ish API:
    logits, loss, new_kvs = model(idx, targets=None, pos=None, past_kvs=None)
- Your model also has model.generate(idx, max_new_tokens), but this file uses an
  instrumented equivalent for the no-cache path so it can measure time-to-first-token.
- Globals such as `device` and `block_size` may already exist in your script.

Suggested use:
1. Paste this near the bottom of your nanoGPT file after training/loading the model.
2. Call `run_baseline_vs_kv_benchmark(m, prompt_tokens, N=64)`.
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
    def tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_tokens / self.total_seconds


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_next_token(logits, temperature=1.0, generator=None):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_no_cache_instrumented(
    model,
    idx,
    max_new_tokens,
    *,
    block_size,
    temperature=1.0,
    generator=None,
):
    """
    Instrumented equivalent of nanoGPT's model.generate().

    It recomputes the full cropped context on every token:
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)

    Returns:
        output_idx: original + generated tokens
        timing: GenerationTiming
    """
    model.eval()

    first_token_seconds = None

    _sync_if_cuda()
    start = time.perf_counter()

    for token_i in range(max_new_tokens):
        step_start = time.perf_counter()

        idx_cond = idx[:, -block_size:]
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = _sample_next_token(
            logits,
            temperature=temperature,
            generator=generator,
        )
        idx = torch.cat((idx, idx_next), dim=1)

        _sync_if_cuda()

        if token_i == 0:
            first_token_seconds = time.perf_counter() - step_start

    _sync_if_cuda()
    total_seconds = time.perf_counter() - start

    return idx, GenerationTiming(
        name="no_cache",
        total_tokens=max_new_tokens,
        total_seconds=total_seconds,
        time_to_first_token=first_token_seconds or 0.0,
    )


@torch.no_grad()
def generate_with_kv_cache_instrumented(
    model,
    idx,
    max_new_tokens,
    *,
    device,
    temperature=1.0,
    generator=None,
):
    """
    Minimal cached generation loop.

    The cache invariant here is simple:
    - after prefill, `past_kvs` contains the full prompt
    - each decode step feeds only the most recently generated token
    - model returns an extended cache

    This assumes your model's cached position handling is already correct.
    """
    model.eval()

    B, prompt_len = idx.shape
    assert B == 1, "This first benchmark is intentionally single-request only."
    assert prompt_len > 0, "Prompt must contain at least one token."

    first_token_seconds = None

    _sync_if_cuda()
    start = time.perf_counter()

    # Prefill prompt once.
    prefill_start = time.perf_counter()
    positions = torch.arange(prompt_len, device=device).unsqueeze(0)
    logits, _, past_kvs = model(idx, pos=positions)

    next_token = _sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        generator=generator,
    )
    idx = torch.cat((idx, next_token), dim=1)

    _sync_if_cuda()
    first_token_seconds = time.perf_counter() - prefill_start

    # Decode remaining tokens one at a time using the cache.
    for _ in range(1, max_new_tokens):
        cache_len = past_kvs[0][0][0].shape[1]
        pos = torch.tensor([[cache_len]], device=device)

        logits, _, past_kvs = model(
            next_token,
            pos=pos,
            past_kvs=past_kvs,
        )

        next_token = _sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            generator=generator,
        )
        idx = torch.cat((idx, next_token), dim=1)

    _sync_if_cuda()
    total_seconds = time.perf_counter() - start

    return idx, GenerationTiming(
        name="kv_cache",
        total_tokens=max_new_tokens,
        total_seconds=total_seconds,
        time_to_first_token=first_token_seconds,
    )


def print_benchmark_table(rows):
    headers = [
        "method",
        "tokens",
        "wall_time_s",
        "tokens_per_s",
        "ttft_ms",
    ]

    table_rows = []
    for row in rows:
        table_rows.append([
            row.name,
            str(row.total_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.tokens_per_second:.2f}",
            f"{row.time_to_first_token * 1000:.2f}",
        ])

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in table_rows))
        for i in range(len(headers))
    ]

    def fmt(values):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
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
    device=None,
    block_size=None,
    seed=1337,
    temperature=1.0,
):
    """
    Run the first benchmark:
    - no-cache generation
    - cached generation
    - wall time
    - tokens/sec
    - time-to-first-token

    `prompt_tokens` should be a Python list of token IDs.
    """
    if device is None:
        device = next(model.parameters()).device

    if block_size is None:
        # If your script has a global block_size, pass it explicitly.
        raise ValueError("Pass block_size explicitly, e.g. block_size=32.")

    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Use separate generators with the same seed so the sampling stream starts
    # from the same state in each method. Outputs may still differ because the
    # model execution order differs, but this keeps runs reproducible.
    gen_no_cache = torch.Generator(device=device)
    gen_no_cache.manual_seed(seed)

    gen_kv = torch.Generator(device=device)
    gen_kv.manual_seed(seed)

    _, no_cache_timing = generate_no_cache_instrumented(
        model,
        prompt.clone(),
        N,
        block_size=block_size,
        temperature=temperature,
        generator=gen_no_cache,
    )

    _, kv_timing = generate_with_kv_cache_instrumented(
        model,
        prompt.clone(),
        N,
        device=device,
        temperature=temperature,
        generator=gen_kv,
    )

    print_benchmark_table([no_cache_timing, kv_timing])

    return {
        "no_cache": no_cache_timing,
        "kv_cache": kv_timing,
    }


# Example call inside your nanoGPT script after training/loading:
#
# context = torch.zeros((1,), dtype=torch.long).tolist()
# run_baseline_vs_kv_benchmark(
#     m,
#     context,
#     N=64,
#     device=device,
#     block_size=block_size,
# )
