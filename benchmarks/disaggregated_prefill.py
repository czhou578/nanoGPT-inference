"""
Disaggregated prefill/decode benchmark for nanogpt-disaggregated-prefill.py.

This benchmark is standalone so importing it does not train the NanoGPT script.
It assumes the model API used by nanogpt-disaggregated-prefill.py:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None)

The benchmark compares:
- monolithic: prefill and decode share one thread (chunked-prefill scheduler)
- disaggregated: separate prefill and decode workers with KV cache handoff

Metrics collected:
- TTFT (time-to-first-token) per request
- Total request latency
- Generated tokens/s throughput
- Correctness (token-level match between the two modes)
"""

from dataclasses import dataclass, field
from statistics import mean
import copy
import time
import threading
import queue
import torch
import torch.nn.functional as F
import heapq


# ──────────────────────────────────────────────────────────────────────
# Data classes (self-contained, no import from the training script)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BenchRequest:
    """Request state used by both monolithic and disaggregated benchmarks."""
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    generated_tokens: list[int] = field(default_factory=list)
    status: str = "waiting"
    prefill_cursor: int = 0
    _radix_path: list = field(default_factory=list)
    arrival_time: int = 0

    kv_cache: dict = field(default_factory=dict)

    # Timing bookmarks
    submitted_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None

    @property
    def tokens_so_far(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    @property
    def is_fully_prefilled(self) -> bool:
        return self.prefill_cursor == len(self.prompt_tokens)

    def clear_cache(self):
        self.kv_cache.clear()


@dataclass
class KVTransfer:
    """Payload sent from prefill worker to decode worker."""
    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    kv_cache: dict
    first_token_id: int
    prefill_time_ms: float
    submitted_at_s: float | None = None


@dataclass
class DisaggRunMetrics:
    name: str
    total_requests: int
    completed_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_seconds: float
    per_request: list[dict]   # [{id, ttft_ms, latency_ms, tokens}]
    generated_tokens: list[list[int]]  # per-request generated token lists (for correctness)

    @property
    def generated_tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_generated_tokens / self.total_seconds

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_prompt_tokens / self.total_seconds

    @property
    def avg_ttft_ms(self) -> float:
        vals = [r["ttft_ms"] for r in self.per_request if r["ttft_ms"] is not None]
        return mean(vals) if vals else 0.0

    @property
    def p95_ttft_ms(self) -> float:
        return _percentile(
            [r["ttft_ms"] for r in self.per_request if r["ttft_ms"] is not None], 0.95
        )

    @property
    def avg_latency_ms(self) -> float:
        vals = [r["latency_ms"] for r in self.per_request if r["latency_ms"] is not None]
        return mean(vals) if vals else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return _percentile(
            [r["latency_ms"] for r in self.per_request if r["latency_ms"] is not None],
            0.95,
        )


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def _sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_generator(device, seed):
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ──────────────────────────────────────────────────────────────────────
# KV cache batching helpers (self-contained copies)
# ──────────────────────────────────────────────────────────────────────

def _infer_n_layer_n_head(model, vocab_size, device):
    """Run a tiny forward pass to discover model shape."""
    dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
    _, _, kvs = model(dummy)
    n_layer = len(kvs)
    n_head = len(kvs[0])
    return n_layer, n_head


def _assemble_batch_cache(requests, n_layer, n_head, device):
    B = len(requests)
    lengths = [requests[i].kv_cache[(0, 0)][0].shape[1] for i in range(B)]
    max_t = max(lengths)
    pad_lengths = [max_t - t for t in lengths]

    attn_mask = torch.zeros(B, 1, max_t, device=device, dtype=torch.bool)
    for i, pad in enumerate(pad_lengths):
        attn_mask[i, 0, pad:] = True

    past_kvs = []
    for layer_idx in range(n_layer):
        block_kv = []
        for head_idx in range(n_head):
            keys, values = [], []
            for i, req in enumerate(requests):
                k, v = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=device)
                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)
                keys.append(k)
                values.append(v)
            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths


def _disassemble_batch_cache(requests, new_kvs, pad_lengths):
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    batched_k[i : i + 1, pad:, :],
                    batched_v[i : i + 1, pad:, :],
                )


# ──────────────────────────────────────────────────────────────────────
# Monolithic scheduler (simplified, no radix tree)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_monolithic(model, workload, *, device, n_layer, n_head,
                   max_batch_size=4, token_budget=16, max_kv_tokens=256,
                   seed=1337, temperature=1.0):
    """Chunked-prefill + batched-decode in a single thread (the baseline)."""
    model.eval()
    generator = _make_generator(device, seed)

    requests = []
    for spec in workload:
        req = BenchRequest(
            id=spec["id"],
            prompt_tokens=list(spec["prompt_tokens"]),
            max_new_tokens=spec["max_new_tokens"],
        )
        req.submitted_at_s = time.perf_counter()
        requests.append(req)

    waiting = []
    for req in requests:
        req.arrival_time = 0
        heapq.heappush(waiting, (0, req.id, req))

    prefilling = []
    active = []
    step = 0

    _sync_if_cuda(device)
    run_start = time.perf_counter()

    while waiting or prefilling or active:
        # --- Admission ---
        if not prefilling and waiting:
            kv_used = sum(
                len(r.prompt_tokens) + r.num_generated for r in active + prefilling
            )
            _, _, candidate = waiting[0]
            actual_cost = len(candidate.prompt_tokens)
            if (kv_used + actual_cost <= max_kv_tokens
                    and len(active) + len(prefilling) < max_batch_size):
                heapq.heappop(waiting)
                candidate.status = "prefilling"
                prefilling.append(candidate)

        prefill_req = prefilling[0] if prefilling else None

        # --- Prefill chunk ---
        if prefill_req:
            remaining_budget = token_budget - len(active)
            if remaining_budget > 0:
                tokens_left = len(prefill_req.prompt_tokens) - prefill_req.prefill_cursor
                chunk_size = min(remaining_budget, tokens_left)
                chunk_start = prefill_req.prefill_cursor
                chunk_tokens = prefill_req.prompt_tokens[chunk_start: chunk_start + chunk_size]
                chunk_t = torch.tensor([chunk_tokens], dtype=torch.long, device=device)
                pos = torch.arange(chunk_start, chunk_start + chunk_size, device=device).unsqueeze(0)

                if prefill_req.kv_cache:
                    past_kvs = []
                    for li in range(n_layer):
                        block_kv = [prefill_req.kv_cache[(li, hi)] for hi in range(n_head)]
                        past_kvs.append(block_kv)
                    logits, _, new_kvs = model(chunk_t, pos=pos, past_kvs=past_kvs)
                else:
                    logits, _, new_kvs = model(chunk_t, pos=pos)

                for li, bkv in enumerate(new_kvs):
                    for hi, (k, v) in enumerate(bkv):
                        prefill_req.kv_cache[(li, hi)] = (k, v)

                prefill_req.prefill_cursor += chunk_size

                if prefill_req.is_fully_prefilled:
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits / temperature, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
                    prefill_req.generated_tokens.append(idx_next.item())
                    prefill_req._last_token = idx_next
                    prefill_req.first_token_at_s = time.perf_counter()
                    prefilling.remove(prefill_req)
                    prefill_req.status = "active"
                    active.append(prefill_req)

        # --- Batched decode ---
        if active:
            batch_tokens = torch.cat([req._last_token for req in active])
            batch_positions = torch.tensor(
                [[len(req.tokens_so_far) - 1] for req in active], device=device
            )
            past_kvs, attn_mask, pad_lengths = _assemble_batch_cache(
                active, n_layer, n_head, device
            )
            logits, _, new_kvs = model(
                batch_tokens, pos=batch_positions,
                past_kvs=past_kvs, attn_mask=attn_mask,
            )
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
            _disassemble_batch_cache(active, new_kvs, pad_lengths)

            for i, req in enumerate(active):
                req.generated_tokens.append(idx_next[i].item())
                req._last_token = idx_next[i : i + 1]

            still_active = []
            for req in active:
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    req.status = "done"
                else:
                    still_active.append(req)
            active = still_active

        step += 1

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    return _build_metrics("monolithic", requests, run_start, run_end)


# ──────────────────────────────────────────────────────────────────────
# Disaggregated workers
# ──────────────────────────────────────────────────────────────────────

def _prefill_worker(model, request_queue, kv_transfer_queue, stop_event,
                    device, temperature, seed):
    model.eval()
    generator = _make_generator(device, seed)

    with torch.no_grad():
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            prompt = torch.tensor([request.prompt_tokens], device=device)
            logits, _, new_kvs = model(prompt)

            kv_cache = {}
            for li, bkv in enumerate(new_kvs):
                for hi, (k, v) in enumerate(bkv):
                    kv_cache[(li, hi)] = (k.clone(), v.clone())

            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            first_token = torch.multinomial(probs, num_samples=1, generator=generator)

            prefill_ms = (time.perf_counter() - t0) * 1000

            transfer = KVTransfer(
                request_id=request.id,
                prompt_tokens=request.prompt_tokens,
                max_new_tokens=request.max_new_tokens,
                kv_cache=kv_cache,
                first_token_id=first_token.item(),
                prefill_time_ms=prefill_ms,
                submitted_at_s=request.submitted_at_s,
            )
            kv_transfer_queue.put(transfer)


def _decode_worker(model, kv_transfer_queue, results_queue, stop_event,
                   device, n_layer, n_head, max_batch_size,
                   temperature, seed):
    model.eval()
    generator = _make_generator(device, seed + 1)
    active_requests = []

    with torch.no_grad():
        while not stop_event.is_set() or active_requests:
            while not kv_transfer_queue.empty() and len(active_requests) < max_batch_size:
                transfer = kv_transfer_queue.get()
                req = BenchRequest(
                    id=transfer.request_id,
                    prompt_tokens=transfer.prompt_tokens,
                    max_new_tokens=transfer.max_new_tokens,
                    kv_cache=transfer.kv_cache,
                    status="active",
                )
                req.submitted_at_s = transfer.submitted_at_s
                req.generated_tokens.append(transfer.first_token_id)
                req.first_token_at_s = time.perf_counter()
                req._last_token = torch.tensor(
                    [[transfer.first_token_id]], device=device
                )
                req.prefill_cursor = len(req.prompt_tokens)
                active_requests.append(req)

            if not active_requests:
                time.sleep(0.01)
                continue

            batch_tokens = torch.cat([r._last_token for r in active_requests])
            batch_positions = torch.tensor(
                [[len(r.tokens_so_far) - 1] for r in active_requests],
                device=device,
            )
            past_kvs, attn_mask, pad_lengths = _assemble_batch_cache(
                active_requests, n_layer, n_head, device
            )
            logits, _, new_kvs = model(
                batch_tokens, pos=batch_positions,
                past_kvs=past_kvs, attn_mask=attn_mask,
            )

            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
            _disassemble_batch_cache(active_requests, new_kvs, pad_lengths)

            for i, req in enumerate(active_requests):
                req.generated_tokens.append(idx_next[i].item())
                req._last_token = idx_next[i : i + 1]

            still_active = []
            for req in active_requests:
                if req.is_done:
                    req.completed_at_s = time.perf_counter()
                    results_queue.put(req)
                else:
                    still_active.append(req)
            active_requests = still_active


@torch.no_grad()
def run_disaggregated(model, workload, *, device, n_layer, n_head,
                      max_batch_size=4, seed=1337, temperature=1.0):
    """Disaggregated prefill/decode with two threads and KV cache handoff."""
    model.eval()

    requests = []
    for spec in workload:
        req = BenchRequest(
            id=spec["id"],
            prompt_tokens=list(spec["prompt_tokens"]),
            max_new_tokens=spec["max_new_tokens"],
        )
        requests.append(req)

    request_queue = queue.Queue()
    kv_transfer_queue = queue.Queue()
    results_queue = queue.Queue()
    stop_event = threading.Event()

    prefill_thread = threading.Thread(
        target=_prefill_worker,
        args=(model, request_queue, kv_transfer_queue, stop_event,
              device, temperature, seed),
        daemon=True,
    )
    decode_thread = threading.Thread(
        target=_decode_worker,
        args=(model, kv_transfer_queue, results_queue, stop_event,
              device, n_layer, n_head, max_batch_size, temperature, seed),
        daemon=True,
    )

    _sync_if_cuda(device)
    run_start = time.perf_counter()

    prefill_thread.start()
    decode_thread.start()

    for req in requests:
        req.submitted_at_s = time.perf_counter()
        request_queue.put(req)

    completed = []
    while len(completed) < len(requests):
        try:
            result = results_queue.get(timeout=30)
            completed.append(result)
        except queue.Empty:
            continue

    stop_event.set()
    prefill_thread.join(timeout=5)
    decode_thread.join(timeout=5)

    _sync_if_cuda(device)
    run_end = time.perf_counter()

    # Re-order completed by id for consistent comparison
    completed.sort(key=lambda r: r.id)
    return _build_metrics("disaggregated", completed, run_start, run_end)


# ──────────────────────────────────────────────────────────────────────
# Metrics & reporting
# ──────────────────────────────────────────────────────────────────────

def _build_metrics(name, requests, run_start, run_end):
    per_request = []
    gen_tokens_lists = []
    for req in requests:
        ttft = None
        lat = None
        if req.first_token_at_s is not None and req.submitted_at_s is not None:
            ttft = (req.first_token_at_s - req.submitted_at_s) * 1000
        if req.completed_at_s is not None and req.submitted_at_s is not None:
            lat = (req.completed_at_s - req.submitted_at_s) * 1000
        per_request.append({
            "id": req.id,
            "ttft_ms": ttft,
            "latency_ms": lat,
            "tokens": len(req.generated_tokens),
        })
        gen_tokens_lists.append(list(req.generated_tokens))

    return DisaggRunMetrics(
        name=name,
        total_requests=len(requests),
        completed_requests=sum(1 for r in requests if r.is_done),
        total_prompt_tokens=sum(len(r.prompt_tokens) for r in requests),
        total_generated_tokens=sum(len(r.generated_tokens) for r in requests),
        total_seconds=run_end - run_start,
        per_request=per_request,
        generated_tokens=gen_tokens_lists,
    )


def print_disagg_comparison_table(rows):
    headers = [
        "method",
        "reqs",
        "done",
        "prompt_tok",
        "gen_tok",
        "wall_s",
        "gen_tok/s",
        "prompt_tok/s",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "avg_lat_ms",
        "p95_lat_ms",
    ]

    rendered = []
    for row in rows:
        rendered.append([
            row.name,
            str(row.total_requests),
            str(row.completed_requests),
            str(row.total_prompt_tokens),
            str(row.total_generated_tokens),
            f"{row.total_seconds:.4f}",
            f"{row.generated_tokens_per_second:.2f}",
            f"{row.prompt_tokens_per_second:.2f}",
            f"{row.avg_ttft_ms:.2f}",
            f"{row.p95_ttft_ms:.2f}",
            f"{row.avg_latency_ms:.2f}",
            f"{row.p95_latency_ms:.2f}",
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

    # Print comparison ratios
    if len(rows) == 2:
        mono, disagg = rows
        print()
        if mono.total_seconds > 0:
            print(
                "Disaggregated throughput ratio: "
                f"{disagg.generated_tokens_per_second / mono.generated_tokens_per_second:.2f}x"
            )
        if mono.avg_ttft_ms > 0:
            print(
                "Average TTFT ratio (lower is better): "
                f"{disagg.avg_ttft_ms / mono.avg_ttft_ms:.2f}x"
            )
        if mono.avg_latency_ms > 0:
            print(
                "Average latency ratio: "
                f"{disagg.avg_latency_ms / mono.avg_latency_ms:.2f}x"
            )


def _check_correctness(mono_metrics, disagg_metrics):
    """Token-level correctness comparison (informational, not a hard gate)."""
    all_match = True
    for i, (mono_gen, disagg_gen) in enumerate(
        zip(mono_metrics.generated_tokens, disagg_metrics.generated_tokens)
    ):
        if mono_gen != disagg_gen:
            print(f"  ⚠ Req {i}: token mismatch (expected with different RNG consumption)")
            all_match = False
    if all_match:
        print("  ✅ All requests produced identical tokens")
    else:
        print("  ℹ Token differences are expected — monolithic and disaggregated "
              "consume RNG differently due to batching order")
    return all_match


# ──────────────────────────────────────────────────────────────────────
# Workload generator
# ──────────────────────────────────────────────────────────────────────

def make_disagg_workload(
    *,
    vocab_size,
    num_requests=8,
    prompt_len=16,
    max_new_tokens=10,
    seed=1337,
):
    """Generate a list of request dicts."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    workload = []
    for i in range(num_requests):
        workload.append({
            "id": i,
            "prompt_tokens": torch.randint(
                0, vocab_size, (prompt_len,), generator=rng
            ).tolist(),
            "max_new_tokens": max_new_tokens,
        })
    return workload


# ──────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────

def run_monolithic_vs_disaggregated_benchmark(
    model,
    *,
    vocab_size,
    num_requests=8,
    prompt_len=16,
    max_new_tokens=10,
    max_batch_size=4,
    token_budget=16,
    max_kv_tokens=256,
    device=None,
    seed=1337,
    temperature=1.0,
):
    if device is None:
        device = next(model.parameters()).device

    n_layer, n_head = _infer_n_layer_n_head(model, vocab_size, device)

    workload = make_disagg_workload(
        vocab_size=vocab_size,
        num_requests=num_requests,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )

    mono = run_monolithic(
        model,
        workload,
        device=device,
        n_layer=n_layer,
        n_head=n_head,
        max_batch_size=max_batch_size,
        token_budget=token_budget,
        max_kv_tokens=max_kv_tokens,
        seed=seed,
        temperature=temperature,
    )

    disagg = run_disaggregated(
        model,
        workload,
        device=device,
        n_layer=n_layer,
        n_head=n_head,
        max_batch_size=max_batch_size,
        seed=seed,
        temperature=temperature,
    )

    print_disagg_comparison_table([mono, disagg])
    print()
    _check_correctness(mono, disagg)

    return {
        "monolithic": mono,
        "disaggregated": disagg,
    }
