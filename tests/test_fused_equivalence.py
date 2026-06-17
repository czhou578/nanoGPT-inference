"""
Equivalence test: fused CausalSelfAttention vs. old Head + MultiHeadAttention.

Verifies that the fused QKV projection produces identical logits to the
separate-head architecture, given the same weights.

Key insight being tested:
  The old model stores Q/K/V as 12 separate (head_size, n_embd) matrices.
  The new model fuses them into one (3*n_embd, n_embd) matrix, ordered:
      [Q_head0, Q_head1, ..., K_head0, K_head1, ..., V_head0, V_head1, ...]

Run:
    python tests/test_fused_equivalence.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Shared hyperparameters ─────────────────────────────────────────────────────
n_embd    = 32
n_head    = 4
n_layer   = 4
head_size = n_embd // n_head   # 8
block_size = 64
vocab_size = 65
dropout   = 0.0
device    = "cpu"

# ── Old architecture: Head + MultiHeadAttention ────────────────────────────────

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# ── New architecture: CausalSelfAttention ─────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.qkv       = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attn_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.num_heads = num_heads
        self.head_size = head_size
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.key_cache   = None
        self.value_cache = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        scale = self.head_size ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale
        attn  = attn.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)
        out   = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_proj(out)


# ── Weight conversion ──────────────────────────────────────────────────────────

def convert_weights(old_mha: MultiHeadAttention, new_csa: CausalSelfAttention):
    """
    Copy weights from old MultiHeadAttention into new CausalSelfAttention.

    The fused qkv weight is laid out as:
        rows  0 ..  n_embd-1          → all Q heads stacked  (Q0, Q1, Q2, Q3)
        rows  n_embd .. 2*n_embd-1    → all K heads stacked  (K0, K1, K2, K3)
        rows  2*n_embd .. 3*n_embd-1  → all V heads stacked  (V0, V1, V2, V3)

    This matches the split(n_embd, dim=2) call in forward().
    """
    with torch.no_grad():
        q_weights = torch.cat([h.query.weight for h in old_mha.heads], dim=0)  # (n_embd, n_embd)
        k_weights = torch.cat([h.key.weight   for h in old_mha.heads], dim=0)
        v_weights = torch.cat([h.value.weight for h in old_mha.heads], dim=0)

        new_csa.qkv.weight.copy_(torch.cat([q_weights, k_weights, v_weights], dim=0))
        new_csa.attn_proj.weight.copy_(old_mha.proj.weight)


# ── Test ───────────────────────────────────────────────────────────────────────

def test_equivalence():
    torch.manual_seed(42)
    old_mha = MultiHeadAttention(n_head, head_size).eval()

    torch.manual_seed(99)   # different seed — weights will be overwritten anyway
    new_csa = CausalSelfAttention(n_head, head_size).eval()

    convert_weights(old_mha, new_csa)

    # Same input, no gradient needed
    torch.manual_seed(0)
    x = torch.randn(2, 16, n_embd)   # (B=2, T=16, C=32)

    with torch.no_grad():
        old_out = old_mha(x)
        new_out = new_csa(x)

    max_diff = (old_out - new_out).abs().max().item()
    print(f"Max absolute difference: {max_diff:.2e}")

    assert torch.allclose(old_out, new_out, atol=1e-6), (
        f"FAIL — outputs differ by up to {max_diff:.2e}"
    )
    print("PASS — fused attention is numerically equivalent to separate-head attention.")


if __name__ == "__main__":
    test_equivalence()
