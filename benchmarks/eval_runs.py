"""
Eval harness runner — runs quality evaluation across NanoGPT implementations.

This is a standalone script that:
  1. Builds a small model and trains it briefly on Tiny Shakespeare
  2. Runs the eval harness against multiple generate functions
  3. Generates/loads a frozen baseline
  4. Compares each implementation against the baseline for regressions
  5. Saves all results to results/eval_results.json

Each generate function is tested through the benchmark helper layer —
the same functions used by the existing correctness equivalence tests.

Usage:
    python benchmarks/eval_runs.py

    # Or from another script:
    from benchmarks.eval_runs import run_eval_suite
    results = run_eval_suite()
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.nn import functional as F

from benchmarks.eval_harness import (
    EvalHarness,
    EvalResult,
    print_eval_result,
    print_regression_report,
    print_comparison_table,
)


# ──────────────────────────────────────────────────────────────────────
# Model architecture (self-contained, matches nanogpt-kv-cache.py)
# ──────────────────────────────────────────────────────────────────────
# These are intentionally duplicated from the implementations so that
# importing this file does NOT trigger training from nanogpt-kv-cache.py.
# The architecture matches what the benchmark helpers expect.

# Small hyperparameters for fast CPU eval
BATCH_SIZE = 8
BLOCK_SIZE = 64
MAX_ITERS = 120
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
# Generate functions (one per "implementation" we want to eval)
# ──────────────────────────────────────────────────────────────────────

def _clear_kv_cache(model):
    """Clear KV caches in all Head modules."""
    for module in model.modules():
        if hasattr(module, "key_cache"):
            module.key_cache = None
        if hasattr(module, "value_cache"):
            module.value_cache = None


def generate_baseline(model, idx, max_new_tokens):
    """
    Vanilla autoregressive — full recompute every step (no KV cache).
    Matches nanogpt.py's model.generate().
    """
    model.train()  # Disables the KV cache branch in Head
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = model(idx_cond, start_pos=0)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_kv_cache(model, idx, max_new_tokens):
    """
    KV-cached generation: prefill once, then decode one token at a time.
    Matches nanogpt-kv-cache.py's generate_kv_cache().
    """
    model.eval()
    _clear_kv_cache(model)

    with torch.no_grad():
        # Prefill
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Decode
        for _ in range(max_new_tokens - 1):
            start_pos = idx.shape[1] - 1
            logits, _ = model(idx[:, -1:], start_pos=start_pos)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx


def generate_feed_one_token(model, idx, max_new_tokens):
    """
    Feed-one-token cached — starts cache from scratch, feeds tokens one by one.
    This simulates what generate_with_cache() does in nanogpt-kv-cache.py.
    """
    model.eval()
    _clear_kv_cache(model)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -1:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx


def generate_greedy_baseline(model, idx, max_new_tokens):
    """
    Deterministic greedy baseline — argmax instead of sampling.
    Used as the consistency reference since it's perfectly reproducible.
    """
    model.train()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = model(idx_cond, start_pos=0)
            logits = logits[:, -1, :]
            idx_next = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_greedy_kv_cache(model, idx, max_new_tokens):
    """
    Deterministic greedy + KV cache — argmax decoding, no sampling.
    """
    model.eval()
    _clear_kv_cache(model)

    with torch.no_grad():
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        idx_next = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

        for _ in range(max_new_tokens - 1):
            start_pos = idx.shape[1] - 1
            logits, _ = model(idx[:, -1:], start_pos=start_pos)
            logits = logits[:, -1, :]
            idx_next = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def _load_data():
    """Load and encode Tiny Shakespeare."""
    input_path = Path(__file__).resolve().parent.parent / "input.txt"
    if not input_path.exists():
        # Fallback: try from CWD
        input_path = Path("input.txt")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find input.txt. Expected at {input_path}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size


def _get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


def _train_model(model, train_data, val_data, verbose=True):
    """Train the model briefly for eval purposes."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iteration in range(MAX_ITERS):
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            if verbose:
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
                print(
                    f"  step {iteration}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

        xb, yb = _get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# ──────────────────────────────────────────────────────────────────────
# Main eval suite
# ──────────────────────────────────────────────────────────────────────

def run_eval_suite(
    *,
    num_prompts: int = 20,
    prompt_len: int = 16,
    max_new_tokens: int = 30,
    num_ppl_windows: int = 50,
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the full eval suite:
      1. Train a small model
      2. Evaluate each generate variant
      3. Compare against baseline
      4. Save results

    Returns dict mapping implementation name → EvalResult.
    """
    print("\n" + "=" * 60)
    print("  NanoGPT Eval Harness")
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

    # Define generate functions to evaluate
    implementations = {
        "baseline_no_cache": generate_baseline,
        "kv_cache_prefill_decode": generate_kv_cache,
        "kv_cache_feed_one": generate_feed_one_token,
        "greedy_no_cache": generate_greedy_baseline,
        "greedy_kv_cache": generate_greedy_kv_cache,
    }

    # Run eval
    if verbose:
        print(f"\n  Running eval harness across {len(implementations)} implementations...")

    harness = EvalHarness(
        model,
        val_data,
        vocab_size,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        model_returns_3=False,  # This model returns (logits, loss)
    )

    results = {}

    for name, gen_fn in implementations.items():
        if verbose:
            print(f"\n  {'─' * 40}")
            print(f"  Evaluating: {name}")

        result = harness.run_full_eval(
            gen_fn,
            name,
            num_prompts=num_prompts,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            num_ppl_windows=num_ppl_windows,
        )

        results[name] = result
        if verbose:
            print_eval_result(result)

    # Print comparison table
    print_comparison_table(list(results.values()))

    # Compare all implementations against the baseline
    baseline = results.get("baseline_no_cache")
    if baseline is not None:
        print("\n" + "=" * 60)
        print("  Regression Checks (vs baseline_no_cache)")
        print("=" * 60)

        for name, result in results.items():
            if name == "baseline_no_cache":
                continue
            report = EvalHarness.compare_to_baseline(result, baseline)
            print_regression_report(report)

    # Save results
    if save_results:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)

        # Save all results
        all_results_path = results_dir / "eval_results.json"
        all_results_dict = {name: r.to_dict() for name, r in results.items()}
        with open(all_results_path, "w") as f:
            json.dump(all_results_dict, f, indent=2)
        if verbose:
            print(f"\n  📁 Saved results to {all_results_path}")

        # Save/update baseline
        baseline_path = results_dir / "eval_baseline.json"
        if baseline is not None and not baseline_path.exists():
            baseline.to_json(str(baseline_path))
            if verbose:
                print(f"  📁 Saved baseline to {baseline_path}")
        elif baseline_path.exists() and verbose:
            print(f"  📁 Baseline already exists at {baseline_path}")
            print(f"      Delete it to regenerate from current run.")

    return results


# ──────────────────────────────────────────────────────────────────────
# Regression detection entry point
# ──────────────────────────────────────────────────────────────────────

def check_regressions(
    results: dict[str, EvalResult],
    baseline_path: str | None = None,
) -> bool:
    """
    Check all results against a frozen baseline file.

    Returns True if all pass, False if any regression detected.
    Designed to be called from CI or a test runner.
    """
    if baseline_path is None:
        baseline_path = str(
            Path(__file__).resolve().parent.parent / "results" / "eval_baseline.json"
        )

    if not Path(baseline_path).exists():
        print(f"  ⚠ No baseline found at {baseline_path}")
        print(f"    Run `python benchmarks/eval_runs.py` first to generate one.")
        return True  # Can't check without a baseline

    baseline = EvalResult.from_json(baseline_path)
    all_pass = True

    for name, result in results.items():
        report = EvalHarness.compare_to_baseline(result, baseline)
        if not report.overall_pass:
            print_regression_report(report)
            all_pass = False

    if all_pass:
        print("  ✅ All implementations pass regression checks.")

    return all_pass


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_eval_suite()

    # Exit with non-zero code if any regression detected
    baseline_path = Path(__file__).resolve().parent.parent / "results" / "eval_baseline.json"
    if baseline_path.exists():
        all_pass = check_regressions(results, str(baseline_path))
        sys.exit(0 if all_pass else 1)
