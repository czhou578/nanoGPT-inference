import torch, time

def naive_attention(q, k, v, causal=False):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d**0.5
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v

def bench(fn, *args, n=20):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n): fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) / n, torch.cuda.max_memory_allocated() / 1e9


for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
    q, k, v = (torch.randn(1, 8, seq_len, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    t_naive, m_naive = bench(naive_attention, q, k, v, True)
    t_sdpa, m_sdpa = bench(lambda q,k,v: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), q, k, v)
    print(f"seq_len={seq_len:6d}  naive: {t_naive*1e3:.2f}ms {m_naive:.2f}GB   sdpa: {t_sdpa*1e3:.2f}ms {m_sdpa:.2f}GB")


def flash_attention_toy(Q, K, V, block_size=128, causal=False):
    seq_len, d = Q.shape
    O = torch.zeros_like(Q)
    for i in range(0, seq_len, block_size):
        Qi = Q[i:i+block_size]
        Oi = torch.zeros_like(Qi)
        li = torch.zeros(Qi.shape[0], 1)
        mi = torch.full((Qi.shape[0], 1), float('-inf'))

        for j in range(0, seq_len, block_size):
            if causal and j > i + block_size - 1:
                break
            Kj, Vj = K[j:j+block_size], V[j:j+block_size]
            Sij = Qi @ Kj.T / d**0.5
            if causal:
                row = torch.arange(i, i+Qi.shape[0]).unsqueeze(1)
                col = torch.arange(j, j+Kj.shape[0]).unsqueeze(0)
                Sij = Sij.masked_fill(col > row, float('-inf'))

            m_new = torch.maximum(mi, Sij.max(dim=-1, keepdim=True).values)
            P_ij = torch.exp(Sij - m_new)
            alpha = torch.exp(mi - m_new)          # rescale factor for old stats

            li = alpha * li + P_ij.sum(dim=-1, keepdim=True)
            Oi = alpha * Oi + P_ij @ Vj
            mi = m_new

        O[i:i+Qi.shape[0]] = Oi / li
        
    return O