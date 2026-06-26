"""
Generate example trace files from profiled inference runs.

Trains a small model from scratch (same as eval_runs.py) and runs the
profiled interleaved generate with a set of requests. Produces JSON
trace files in Chrome Trace Event Format.

Usage:
    python -m profiler.generate_traces

Output:
    profiler/traces/trace_interleaving.json
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from profiler import trace
from profiler.profiled_engine import (
    configure,
    GPTLanguageModel,
    Request,
    interleaved_generate_profiled,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration (matches the small training config in the .py files)
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 8
BLOCK_SIZE = 64
MAX_ITERS = 120
EVAL_INTERVAL = 20
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 10
N_EMBD = 32
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0

torch.manual_seed(1337)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input.txt")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model():
    """Train a small model from a fixed seed. Returns the model in eval mode."""
    configure(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
        device=DEVICE,
    )

    model = GPTLanguageModel().to(DEVICE)
    print(f"  Model: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M parameters")
    print(f"  Device: {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(f"  step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Trace Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_interleaving_trace(model):
    """
    Run the profiled interleaved generate with 4 staggered requests.

    Produces a trace showing:
      - Scheduler decisions at each step
      - Fused batch assembly (decode + prefill packed together)
      - Forward pass timing
      - Cache disassembly
      - Sampling
      - Request lifecycle events (admitted, prefill complete, done)
    """
    # Create requests with varying prompt lengths for an interesting trace
    prompts = [
        "First Citizen:",
        "ROMEO: But soft,",
        "To be or not",
        "All that glitters",
    ]

    requests = []
    for i, prompt_text in enumerate(prompts):
        tokens = encode(prompt_text)
        requests.append(Request(
            id=i,
            prompt_tokens=tokens,
            max_new_tokens=12,
            priority=i % 3,
        ))

    print(f"\n  Requests:")
    for req in requests:
        print(f"    req_{req.id}: prompt_len={len(req.prompt_tokens)}, "
              f"max_new={req.max_new_tokens}, priority={req.priority}")

    # Run with profiling enabled
    trace.begin_trace()

    scheduler = interleaved_generate_profiled(
        model, requests,
        policy="fcfs",
        token_budget=16,
        max_kv_tokens=128,
    )

    trace.end_trace()

    # Print results
    print(f"\n  Generation results:")
    for req in requests:
        text = decode(req.tokens_so_far)
        print(f"    req_{req.id}: {repr(text[:60])}")

    # Print summary
    summary = trace.summary()
    print(f"\n  Trace summary:")
    print(f"    Total time: {summary['total_us'] / 1000:.1f} ms")
    print(f"    Total spans: {summary['total_spans']}")
    print(f"    Categories:")
    for cat, info in summary["by_category"].items():
        print(f"      {cat:20s}: {info['total_us'] / 1000:6.1f} ms "
              f"({info['pct']:4.1f}%) [{info['count']} spans]")

    return summary


def main():
    print("=" * 60)
    print("  Inference Profiler - Trace Generation")
    print("=" * 60)

    # 1. Train the model
    print("\n  Training model...")
    model = train_model()

    # 2. Generate traces
    traces_dir = os.path.join(os.path.dirname(__file__), "traces")
    os.makedirs(traces_dir, exist_ok=True)

    # Interleaving trace
    print("\n  Generating interleaving trace...")
    summary = generate_interleaving_trace(model)

    trace_path = os.path.join(traces_dir, "trace_interleaving.json")
    trace.export_json(
        trace_path,
        implementation="interleaved_generate",
        model_params=sum(p.numel() for p in model.parameters()),
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        device=DEVICE,
    )

    print(f"\n  Trace file: {trace_path}")
    print(f"  Open in chrome://tracing or https://ui.perfetto.dev")
    print("=" * 60)


if __name__ == "__main__":
    main()
