"""
Attention Backend Ablation — naive vs SDPA vs pure-PyTorch FA2.

Benchmarks three attention implementations across a grid of
(sequence_length, batch_size) on your actual hardware.  The goal is to
quantify the gap between the O(N²)-memory naive path and PyTorch's
fused SDPA kernel, and to see where the pure-PyTorch FA2 tiled
implementation (which never materializes the full attention matrix)
sits between them.

On the Jetson GB10 (sm_121) the official FlashAttention CUDA kernels
don't compile — SDPA falls back to the C++ "math" path — so this
benchmark documents what you *actually* get on this hardware.

Run:
    python attention-backend-benchmark.py
"""

import math
import time
import argparse

import torch
import torch.nn.functional as F


# ── Backend implementations ──────────────────────────────────────────────────

def naive_attention(Q, K, V, causal=True):
    """
    Textbook attention: materializes the full (N × N) score matrix.
    O(N² · d) compute, O(N²) memory.
    """
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d ** 0.5          # (B, H, N, N)
    if causal:
        N = scores.shape[-1]
        mask = torch.triu(
            torch.ones(N, N, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ V


def sdpa_attention(Q, K, V, causal=True):
    """
    PyTorch's scaled_dot_product_attention — dispatches to whichever
    fused kernel is available (FlashAttention, memory-efficient, or the
    C++ math fallback on unsupported architectures like sm_121).
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def pytorch_fa2_attention(Q, K, V, causal=True, block_size=128):
    """
    Pure-PyTorch tiled FlashAttention-2 forward pass.

    Outer loop over Q blocks, inner loop over K/V blocks, with online
    softmax (running max + sum) so the full N × N matrix is never
    materialised.  This is the FA2 loop order specifically.

    Causal masking is applied per-tile when the K/V block sits at or
    past the diagonal.
    """
    B, H, N, D = Q.shape
    scale = D ** -0.5
    O = torch.zeros_like(Q)
    n_blocks = math.ceil(N / block_size)

    for bi in range(n_blocks):
        i0 = bi * block_size
        i1 = min(i0 + block_size, N)
        Qi = Q[:, :, i0:i1, :]

        Oi = torch.zeros_like(Qi)
        li = torch.zeros(B, H, i1 - i0, 1, device=Q.device, dtype=Q.dtype)
        mi = torch.full(
            (B, H, i1 - i0, 1), float("-inf"), device=Q.device, dtype=Q.dtype
        )

        # In causal mode we only need K/V blocks up to the current Q block
        kv_limit = bi + 1 if causal else n_blocks

        for bj in range(kv_limit):
            j0 = bj * block_size
            j1 = min(j0 + block_size, N)
            Kj = K[:, :, j0:j1, :]
            Vj = V[:, :, j0:j1, :]

            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

            # Apply causal mask on the diagonal tile
            if causal and bi == bj:
                tile_rows = i1 - i0
                tile_cols = j1 - j0
                row_idx = torch.arange(i0, i1, device=Q.device).unsqueeze(1)
                col_idx = torch.arange(j0, j1, device=Q.device).unsqueeze(0)
                causal_mask = col_idx > row_idx          # upper-triangle
                Sij = Sij.masked_fill(causal_mask, float("-inf"))

            m_new = torch.maximum(mi, Sij.amax(dim=-1, keepdim=True))
            P_ij = torch.exp(Sij - m_new)
            alpha = torch.exp(mi - m_new)

            li = alpha * li + P_ij.sum(dim=-1, keepdim=True)
            Oi = alpha * Oi + torch.matmul(P_ij, Vj)
            mi = m_new

        O[:, :, i0:i1, :] = Oi / li

    return O


# ── Registry ─────────────────────────────────────────────────────────────────

BACKENDS = {
    "naive":       naive_attention,
    "sdpa":        sdpa_attention,
    "pytorch_fa2": pytorch_fa2_attention,
}


# ── Correctness check ────────────────────────────────────────────────────────

def check_correctness(device, dtype=torch.float32):
    """All backends should match the naive reference within tolerance."""
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 512, 64
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N, D, device=device, dtype=dtype)

    ref = naive_attention(Q, K, V, causal=True)

    print("Correctness check (causal, fp32, N=512):")
    all_ok = True
    for name, fn in BACKENDS.items():
        out = fn(Q, K, V, causal=True)
        ok = torch.allclose(ref, out, atol=1e-4, rtol=1e-4)
        status = "✓" if ok else "✗"
        if not ok:
            max_diff = (ref - out).abs().max().item()
            print(f"  {status} {name:14s}  max_diff={max_diff:.2e}")
            all_ok = False
        else:
            print(f"  {status} {name}")
    return all_ok


# ── Benchmarking helpers ─────────────────────────────────────────────────────

def bench_one(fn, Q, K, V, *, warmup=5, repeats=20):
    """
    Time a single backend.  Returns (median_ms, peak_memory_gb).
    Uses CUDA events for accurate GPU timing.
    """
    device = Q.device

    # Warmup
    for _ in range(warmup):
        fn(Q, K, V, causal=True)
    torch.cuda.synchronize()

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats(device)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn(Q, K, V, causal=True)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    median_ms = times[len(times) // 2]

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return median_ms, peak_mem_gb


def run_benchmark_grid(
    device,
    dtype,
    seq_lengths,
    batch_sizes,
    n_heads=8,
    head_dim=64,
    backends=None,
):
    """
    Sweep (seq_len × batch_size × backend) and return structured results.

    Returns a list of dicts, one per measurement point.
    """
    if backends is None:
        backends = list(BACKENDS.keys())

    results = []

    for B in batch_sizes:
        for N in seq_lengths:
            # Allocate Q, K, V once per (B, N) combo
            try:
                Q, K, V = (
                    torch.randn(B, n_heads, N, head_dim, device=device, dtype=dtype)
                    for _ in range(3)
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM allocating tensors for B={B}, N={N} — skipping")
                for name in backends:
                    results.append({
                        "backend": name, "batch_size": B, "seq_len": N,
                        "time_ms": float("nan"), "peak_mem_gb": float("nan"),
                        "status": "OOM",
                    })
                continue

            for name in backends:
                fn = BACKENDS[name]
                try:
                    torch.cuda.empty_cache()
                    t_ms, mem_gb = bench_one(fn, Q, K, V)
                    results.append({
                        "backend": name, "batch_size": B, "seq_len": N,
                        "time_ms": t_ms, "peak_mem_gb": mem_gb,
                        "status": "ok",
                    })
                    print(
                        f"  B={B:2d}  N={N:6d}  {name:14s}  "
                        f"{t_ms:8.2f} ms   {mem_gb:.3f} GB"
                    )
                except torch.cuda.OutOfMemoryError:
                    results.append({
                        "backend": name, "batch_size": B, "seq_len": N,
                        "time_ms": float("nan"), "peak_mem_gb": float("nan"),
                        "status": "OOM",
                    })
                    print(f"  B={B:2d}  N={N:6d}  {name:14s}  OOM")
                    torch.cuda.empty_cache()

            # Free tensors before the next (B, N) pair
            del Q, K, V
            torch.cuda.empty_cache()

    return results


# ── Pretty-print results ─────────────────────────────────────────────────────

def print_summary_table(results, backends):
    """
    Print a readable table grouped by batch size, with one row per
    seq_len and columns for each backend's latency.
    """
    from collections import defaultdict

    # Group by (batch_size, seq_len)
    grouped = defaultdict(dict)
    for r in results:
        grouped[(r["batch_size"], r["seq_len"])][r["backend"]] = r

    batch_sizes = sorted({r["batch_size"] for r in results})
    seq_lengths = sorted({r["seq_len"] for r in results})

    # Header
    header_parts = [f"{'N':>8s}"]
    for name in backends:
        header_parts.append(f"{name + ' (ms)':>18s}")
        header_parts.append(f"{'mem (GB)':>10s}")
    header = "  ".join(header_parts)

    for B in batch_sizes:
        print(f"\n{'─' * len(header)}")
        print(f"  Batch size = {B}")
        print(f"{'─' * len(header)}")
        print(header)
        print("─" * len(header))

        for N in seq_lengths:
            key = (B, N)
            if key not in grouped:
                continue
            row_parts = [f"{N:8d}"]
            for name in backends:
                r = grouped[key].get(name)
                if r is None or r["status"] != "ok":
                    row_parts.append(f"{'OOM':>18s}")
                    row_parts.append(f"{'—':>10s}")
                else:
                    row_parts.append(f"{r['time_ms']:18.2f}")
                    row_parts.append(f"{r['peak_mem_gb']:10.3f}")
            print("  ".join(row_parts))

    print()


def save_results(results, backends, filepath):
    """Save the summary table to a text file."""
    import io
    import sys

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_summary_table(results, backends)
    sys.stdout = old_stdout

    with open(filepath, "w") as f:
        f.write(f"Attention Backend Ablation\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"Causal: True, Heads: 8, Head dim: 64, dtype: bfloat16\n\n")
        f.write(buf.getvalue())

    print(f"Results saved to {filepath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention backend ablation benchmark"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["naive", "sdpa", "pytorch_fa2"],
        choices=list(BACKENDS.keys()),
        help="Which backends to benchmark",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to sweep",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="Dimension per head"
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip correctness check",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/attention_backend_results.txt",
        help="Output file path",
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU.")
        return

    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"CC:      sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype:   {dtype}")
    print(f"Heads:   {args.heads}, Head dim: {args.head_dim}")
    print(f"Backends: {args.backends}")
    print(f"Seq lengths: {args.seq_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")

    # ── Correctness ──
    if not args.skip_correctness:
        print()
        ok = check_correctness(device)
        if not ok:
            print("\n⚠  Correctness check failed — results may not be comparable")

    # ── Benchmark ──
    print(f"\nRunning benchmark grid...")
    results = run_benchmark_grid(
        device=device,
        dtype=dtype,
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        n_heads=args.heads,
        head_dim=args.head_dim,
        backends=args.backends,
    )

    # ── Report ──
    print_summary_table(results, args.backends)
    save_results(results, args.backends, args.output)


if __name__ == "__main__":
    main()
