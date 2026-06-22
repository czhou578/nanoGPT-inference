"""
Streaming inference server for NanoGPT with radix-tree prefix caching
and real-time metrics.

Start:
    python server.py

Single request:
    curl -N http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "First Citizen:", "max_tokens": 50}'

Two concurrent requests (separate terminals):
    curl -N http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "ROMEO:", "max_tokens": 30}'
    curl -N http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "JULIET:", "max_tokens": 30}'

Shared prefix (radix cache hit on second request):
    curl -N http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "First Citizen: We are accounted poor", "max_tokens": 20}'
    curl -N http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "First Citizen: We are accounted poor", "max_tokens": 20}'

Check engine state:
    curl localhost:8000/health

Metrics (JSON):
    curl localhost:8000/metrics

Metrics (Prometheus):
    curl -H "Accept: text/plain" localhost:8000/metrics
"""

import asyncio
import importlib.util
import json
import os
import threading
import time

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

from benchmarks.metrics import InferenceMetrics


# ── Load engine module ────────────────────────────────────────────────────────

def _load_engine_module():
    """Import nanogpt-radix-tree.py via importlib (hyphenated filename)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanogpt-radix-tree.py")
    spec = importlib.util.spec_from_file_location("_nanogpt_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

E = _load_engine_module()  # short alias — all engine symbols live here


# ── Train the model ───────────────────────────────────────────────────────────

def _train_model():
    print(f"Training NanoGPT on {E.device}...")
    model = E.GPTLanguageModel().to(E.device)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=E.learning_rate)
    for i in range(E.max_iters):
        xb, yb = E.get_batch("train")
        logits, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % E.eval_interval == 0 or i == E.max_iters - 1:
            print(f"  step {i}: loss {loss.item():.4f}")

    model.eval()
    print("Model trained and ready.\n")
    return model

MODEL = _train_model()


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps the existing Scheduler and runs a continuous generate loop in a
    background thread.  HTTP handlers submit requests via .submit(), which
    returns an asyncio.Queue that receives tokens as they are produced.
    """

    def __init__(self, model, max_batch_size=8, token_budget=64,
                 max_kv_tokens=512, block_size=4):
        self.model = model
        self.scheduler = E.Scheduler(
            policy="fcfs",
            max_batch_size=max_batch_size,
            token_budget=token_budget,
            max_kv_tokens=max_kv_tokens,
            block_size=block_size,
        )
        self.step = 0
        self.metrics = InferenceMetrics()

        # Pending requests (written by HTTP handlers, read by engine thread)
        self._pending: list = []
        self._lock = threading.Lock()

        # Per-request token delivery queues and timing
        self._queues: dict[int, asyncio.Queue] = {}
        self._submit_times: dict[int, float] = {}
        self._last_token_times: dict[int, float] = {}  # for ITL tracking
        self._next_id = 0

        # Set by startup so the engine thread can push to asyncio queues
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Public API (called from async FastAPI handlers) ───────────────────

    def submit(self, prompt_tokens: list[int], max_tokens: int) -> tuple[int, asyncio.Queue]:
        """Create a request and return (request_id, token_queue)."""
        req_id = self._next_id
        self._next_id += 1

        req = E.Request(
            id=req_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
        )
        req.arrival_time = self.step

        queue = asyncio.Queue()
        with self._lock:
            self._pending.append(req)
            self._queues[req_id] = queue
            self._submit_times[req_id] = time.perf_counter()

        self.metrics.inc("requests_total")
        self.metrics.inc("tokens_prefilled_total", len(prompt_tokens))
        self.metrics.inc_gauge("waiting_requests")

        return req_id, queue

    # ── Engine loop (runs in background thread) ───────────────────────────

    def _drain_pending(self):
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
        for req in batch:
            self.scheduler.add_request(req)

    def _put(self, req_id: int, item):
        """Thread-safe push into an asyncio.Queue."""
        q = self._queues.get(req_id)
        if q and self._loop:
            self._loop.call_soon_threadsafe(q.put_nowait, item)

    def _finish(self, req_id: int):
        self._put(req_id, None)  # sentinel
        self._queues.pop(req_id, None)

        # Record E2E latency
        submit_time = self._submit_times.pop(req_id, None)
        if submit_time is not None:
            e2e = time.perf_counter() - submit_time
            self.metrics.observe("e2e_seconds", e2e)

        self._last_token_times.pop(req_id, None)
        self.metrics.inc("requests_completed")
        self.metrics.inc_gauge("active_requests", -1.0)

    def run_loop(self):
        """Main engine loop — mirrors scheduled_generate but never terminates."""
        self.model.eval()

        with torch.no_grad():
            while True:
                self._drain_pending()

                has_work = (
                    self.scheduler.waiting
                    or self.scheduler.prefilling
                    or self.scheduler.active
                )
                if not has_work:
                    time.sleep(0.01)
                    continue

                # Update queue-depth gauges
                self.metrics.set_gauge(
                    "waiting_requests", len(self.scheduler.waiting)
                )
                self.metrics.set_gauge(
                    "active_requests",
                    len(self.scheduler.active) + len(self.scheduler.prefilling),
                )

                prefill_req, decode_reqs = self.scheduler.schedule(self.step)

                # ── Prefill ──────────────────────────────────────────
                if prefill_req:
                    remaining_budget = self.scheduler.token_budget - len(self.scheduler.active)

                    if remaining_budget > 0 and self.scheduler.prefilling:
                        p_req = self.scheduler.prefilling[0]
                        tokens_left = len(p_req.prompt_tokens) - p_req.prefill_cursor
                        chunk_size = min(remaining_budget, tokens_left)
                        chunk_start = p_req.prefill_cursor

                        chunk_tokens = p_req.prompt_tokens[chunk_start:chunk_start + chunk_size]
                        prefill_chunk = torch.tensor(
                            [chunk_tokens], dtype=torch.long, device=E.device
                        )
                        p_req.prefill_cursor += chunk_size

                        pos = torch.arange(
                            chunk_start, chunk_start + chunk_size, device=E.device
                        ).unsqueeze(0)

                        if p_req.kv_cache:
                            past_kvs = []
                            for li in range(E.n_layer):
                                past_kvs.append(
                                    [p_req.kv_cache[(li, hi)] for hi in range(E.n_head)]
                                )
                            logits, _, new_kvs = self.model(
                                prefill_chunk, pos=pos, past_kvs=past_kvs
                            )
                        else:
                            logits, _, new_kvs = self.model(prefill_chunk, pos=pos)

                        for li, bkv in enumerate(new_kvs):
                            for hi, (k, v) in enumerate(bkv):
                                p_req.kv_cache[(li, hi)] = (k, v)

                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)

                        if prefill_req.is_fully_prefilled:
                            E.insert_into_radix_tree(
                                prefill_req,
                                self.scheduler.radix_tree,
                                self.scheduler.block_size,
                            )
                            token_id = idx_next.item()
                            prefill_req.generated_tokens.append(token_id)
                            prefill_req._last_token = idx_next
                            self.scheduler.radix_tree.unlock_radix_path(prefill_req)
                            self.scheduler.promote(prefill_req)

                            # Record TTFT
                            now = time.perf_counter()
                            submit_time = self._submit_times.get(prefill_req.id)
                            if submit_time is not None:
                                ttft = now - submit_time
                                self.metrics.observe("ttft_seconds", ttft)
                            self._last_token_times[prefill_req.id] = now

                            self.metrics.inc("tokens_generated_total")
                            self.metrics.observe(
                                "prefill_tokens_per_step",
                                len(prefill_req.prompt_tokens),
                            )

                            # Stream the first token
                            self._put(prefill_req.id, {
                                "token": E.decode([token_id]),
                                "token_id": token_id,
                                "is_first": True,
                                "ttft_ms": round(
                                    (now - self._submit_times.get(prefill_req.id, 0)) * 1000, 1
                                ),
                            })

                # ── Decode ───────────────────────────────────────────
                if decode_reqs:
                    active = self.scheduler.active
                    batch_tokens = torch.cat([r._last_token for r in active])
                    batch_positions = torch.tensor(
                        [[len(r.tokens_so_far) - 1] for r in active],
                        device=E.device,
                    )

                    past_kvs, attn_mask, pad_lengths = E.assemble_batch_cache(active)

                    logits, _, new_kvs = self.model(
                        batch_tokens,
                        pos=batch_positions,
                        past_kvs=past_kvs,
                        attn_mask=attn_mask,
                    )

                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    E.disassemble_batch_cache(active, new_kvs, pad_lengths)

                    for i, req in enumerate(active):
                        token_id = idx_next[i].item()
                        req.generated_tokens.append(token_id)
                        req._last_token = idx_next[i:i + 1]

                        # Record ITL
                        now = time.perf_counter()
                        last_time = self._last_token_times.get(req.id)
                        if last_time is not None:
                            self.metrics.observe("itl_seconds", now - last_time)
                        self._last_token_times[req.id] = now

                        self.metrics.inc("tokens_generated_total")

                        # Stream the token
                        self._put(req.id, {
                            "token": E.decode([token_id]),
                            "token_id": token_id,
                            "is_first": False,
                        })

                    self.metrics.observe("batch_size_per_step", len(active))

                    for req in list(active):
                        if req.is_done:
                            self.scheduler.radix_tree.unlock_radix_path(req)
                            self.scheduler.complete(req)
                            self._finish(req.id)

                self.step += 1


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(title="NanoGPT Streaming Server")
engine = InferenceEngine(MODEL)


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@app.on_event("startup")
async def startup():
    engine._loop = asyncio.get_event_loop()
    thread = threading.Thread(target=engine.run_loop, daemon=True)
    thread.start()
    print("Engine loop started.")


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    prompt_tokens = E.encode(req.prompt)
    req_id, queue = engine.submit(prompt_tokens, req.max_tokens)

    async def stream():
        full_text = ""
        while True:
            item = await queue.get()
            if item is None:
                yield f"data: {json.dumps({'done': True, 'full_text': full_text})}\n\n"
                break
            full_text += item["token"]
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {
        "step": engine.step,
        "waiting": len(engine.scheduler.waiting),
        "prefilling": len(engine.scheduler.prefilling),
        "active": len(engine.scheduler.active),
        "pending_submit": len(engine._pending),
    }


@app.get("/metrics")
async def metrics(request: Request):
    """Return engine metrics in JSON or Prometheus text format.

    Use Accept: text/plain for Prometheus format, otherwise JSON.
    """
    accept = request.headers.get("accept", "application/json")
    if "text/plain" in accept:
        return PlainTextResponse(
            content=engine.metrics.prometheus_text(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    return engine.metrics.snapshot()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
