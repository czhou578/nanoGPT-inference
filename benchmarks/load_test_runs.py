"""
Load test runner — runs pre-configured load test scenarios on NanoGPT.

This is a standalone script that:
  1. Builds a small model and trains it briefly on Tiny Shakespeare
  2. Runs multiple load test scenarios (light, medium, burst, ramp, poisson)
  3. Prints percentile latency tables for each scenario
  4. Saves all results to results/load_test_results.json

Uses the same model architecture as eval_runs.py, sharing the self-contained
model definition to avoid triggering training from the nanogpt-*.py scripts.

Usage:
    uv run python -m benchmarks.load_test_runs
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from benchmarks.load_tester import LoadTester


# ──────────────────────────────────────────────────────────────────────
# Model architecture (same as eval_runs.py, duplicated for independence)
# ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 4
BLOCK_SIZE = 64
MAX_ITERS = 80
EVAL_INTERVAL = 20
LEARNING_RATE = 1e-3
DEVICE = "cpu"
EVAL_ITERS = 10
N_EMBD = 32
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0


class Head(nn.Module):
    """One head of self-attention with optional KV cache."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.key_cache = None
        self.value_cache = None
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if not self.training:
            if self.key_cache is not None:
                self.key_cache = torch.cat([self.key_cache, k], dim=-2)
                self.value_cache = torch.cat([self.value_cache, v], dim=-2)
            else:
                self.key_cache = k
                self.value_cache = v

            wei = q @ self.key_cache.transpose(-2, -1) * (
                self.key_cache.shape[-1] ** -0.5
            )
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ self.value_cache
            return out
        else:
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos=0):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(start_pos, start_pos + T, device=idx.device)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ──────────────────────────────────────────────────────────────────────
# Data loading and training
# ──────────────────────────────────────────────────────────────────────

def _load_data():
    input_path = Path(__file__).resolve().parent.parent / "input.txt"
    if not input_path.exists():
        input_path = Path("input.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input.txt at {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size


def _get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


def _train_model(model, train_data, val_data, verbose=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for iteration in range(MAX_ITERS):
        if verbose and (iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1):
            model.eval()
            losses = {}
            for split in ["train", "val"]:
                loss_acc = torch.zeros(EVAL_ITERS)
                for k in range(EVAL_ITERS):
                    X, Y = _get_batch(split, train_data, val_data)
                    _, loss = model(X, Y)
                    loss_acc[k] = loss.item()
                losses[split] = loss_acc.mean().item()
            model.train()
            print(f"  step {iteration}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

        xb, yb = _get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# ──────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "light_load",
        "concurrency": 2,
        "num_requests": 6,
        "pattern": "constant",
        "max_tokens": 8,
        "prompt_len": 12,
    },
    {
        "name": "medium_load",
        "concurrency": 4,
        "num_requests": 10,
        "pattern": "constant",
        "max_tokens": 8,
        "prompt_len": 12,
    },
    {
        "name": "burst",
        "concurrency": 6,
        "num_requests": 6,
        "pattern": "burst",
        "max_tokens": 6,
        "prompt_len": 10,
    },
    {
        "name": "ramp_up",
        "concurrency": 4,
        "num_requests": 8,
        "pattern": "ramp",
        "max_tokens": 8,
        "prompt_len": 12,
    },
    {
        "name": "poisson",
        "concurrency": 4,
        "num_requests": 10,
        "pattern": "poisson",
        "max_tokens": 8,
        "prompt_len": 12,
    },
]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run_load_test_suite(
    *,
    scenarios: list[dict] | None = None,
    save_results: bool = True,
    verbose: bool = True,
) -> list:
    """
    Run all load test scenarios.

    Returns list of LatencyReport objects.
    """
    if scenarios is None:
        scenarios = SCENARIOS

    print("\n" + "=" * 60)
    print("  NanoGPT Load Tester")
    print("=" * 60)

    # Load data
    train_data, val_data, vocab_size = _load_data()
    if verbose:
        print(f"\n  Loaded data: {len(train_data)} train, {len(val_data)} val tokens")
        print(f"  Vocab size: {vocab_size}")

    # Build and train model
    torch.manual_seed(1337)
    model = GPTLanguageModel(vocab_size)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if verbose:
        print(f"  Model: {param_count:.3f}M parameters")
        print(f"\n  Training ({MAX_ITERS} steps)...")

    _train_model(model, train_data, val_data, verbose=verbose)

    # Create load tester
    tester = LoadTester(
        model,
        vocab_size=vocab_size,
        device=DEVICE,
        block_size=BLOCK_SIZE,
    )

    # Run scenarios
    all_reports = []

    for scenario in scenarios:
        if verbose:
            print(f"\n  {'─' * 50}")
            print(f"  Running scenario: {scenario['name']} "
                  f"(pattern={scenario['pattern']}, concurrency={scenario['concurrency']})")

        report = tester.run_load_test(
            scenario_name=scenario["name"],
            pattern=scenario["pattern"],
            concurrency=scenario["concurrency"],
            num_requests=scenario["num_requests"],
            prompt_len=scenario.get("prompt_len", 12),
            max_tokens=scenario.get("max_tokens", 8),
            val_data=val_data,
        )

        all_reports.append(report)
        if verbose:
            tester.print_report(report)

    # Summary table
    if verbose:
        tester.print_sweep_summary(all_reports)

    # Save results
    if save_results:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / "load_test_results.json"
        tester.save_results(all_reports, str(results_path))
        if verbose:
            print(f"  📁 Saved results to {results_path}")

    return all_reports


if __name__ == "__main__":
    run_load_test_suite()
