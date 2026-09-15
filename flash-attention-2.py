"""
FlashAttention-2 forward pass — pure PyTorch reference, non-causal.

Every line maps directly to the FA2 algorithm: outer loop over Q blocks,
inner loop over K/V blocks, with a running max/sum (online softmax) so the
full N x N attention matrix is never materialized at once.

This is the FA2 loop order specifically (Q outer, K/V inner) — FA1 loops
K/V outer and Q inner, which needs cross-block accumulation. FA2's ordering
lets each Q block's output be computed independently, which is what made
it roughly 2x faster in practice.
"""

import math
import time

import torch


def flash_attention_2_forward(Q, K, V, block_size=128, softmax_scale=None):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    Returns: (batch, heads, seq_len, head_dim)
    """
    B, H, N, D = Q.shape
    scale = softmax_scale or D ** -0.5
    O = torch.zeros_like(Q)

    n_blocks = math.ceil(N / block_size)

    for bi in range(n_blocks):
        i0, i1 = bi * block_size, min((bi + 1) * block_size, N)
        Qi = Q[:, :, i0:i1, :]

        Oi = torch.zeros_like(Qi)
        li = torch.zeros(B, H, i1 - i0, 1, device=Q.device, dtype=Q.dtype)
        mi = torch.full((B, H, i1 - i0, 1), float("-inf"), device=Q.device, dtype=Q.dtype)

        for bj in range(n_blocks):
            j0, j1 = bj * block_size, min((bj + 1) * block_size, N)
            Kj, Vj = K[:, :, j0:j1, :], V[:, :, j0:j1, :]

            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

            m_new = torch.maximum(mi, Sij.amax(dim=-1, keepdim=True))
            P_ij = torch.exp(Sij - m_new)
            alpha = torch.exp(mi - m_new)  # rescales the *old* accumulator

            li = alpha * li + P_ij.sum(dim=-1, keepdim=True)
            Oi = alpha * Oi + torch.matmul(P_ij, Vj)
            mi = m_new

        O[:, :, i0:i1, :] = Oi / li

    return O


# ---------------------------------------------------------------------------
# Correctness check + benchmark
# ---------------------------------------------------------------------------

def _reference(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


def check_correctness(device="cuda", dtype=torch.float32):
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 1000, 64
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N, D, device=device, dtype=dtype)

    ref = _reference(Q, K, V)
    out = flash_attention_2_forward(Q, K, V, block_size=128)
    ok = torch.allclose(ref, out, atol=1e-4, rtol=1e-4)
    print(f"pure-PyTorch FA2 matches SDPA: {ok}")


def benchmark(device="cuda", dtype=torch.bfloat16):
    print(f"\nBenchmarking on {device}, dtype={dtype}")
    B, H, D = 1, 8, 64
    for N in (512, 1024, 2048, 4096, 8192):
        Q, K, V = (torch.randn(B, H, N, D, device=device, dtype=dtype) for _ in range(3))

        def timeit(fn, n=20):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t0) / n * 1000

        t_sdpa = timeit(lambda: _reference(Q, K, V))
        t_fa2 = timeit(lambda: flash_attention_2_forward(Q, K, V, block_size=128))
        print(f"N={N:6d}  sdpa: {t_sdpa:7.2f}ms   pytorch-fa2: {t_fa2:8.2f}ms")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_correctness(device=device, dtype=torch.float32)
    if device == "cuda":
        benchmark(device=device)