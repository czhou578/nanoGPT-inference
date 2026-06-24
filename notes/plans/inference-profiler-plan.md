# Inference Profiler - Implementation Plan

## The Problem You're Solving

Your benchmark suites measure two things: throughput (tokens/second) and quality (eval harness).
Neither tells you **where time is actually spent** inside the inference pipeline.

When continuous batching is slower than expected, is it because:
- Attention is slow? (compute-bound)
- KV cache assembly takes too long? (memory-bound)
- The scheduler's `_maybe_admit` loop is doing too much work? (overhead)
- `torch.cat` on the cache is allocating and copying? (memory allocation)

You can't answer these questions with wall-clock throughput numbers.
You need per-operation, per-request timing data - and a way to visualize it.

### What this builds

An instrumentation library (`profiler/`) and a React visualization page (`/profiler`) that together produce a flame-chart-style timeline of inference execution.
Every request's journey through the pipeline is visible: when it was admitted, which prefill chunks ran, how long each attention kernel took, when KV blocks were allocated, and how speculative tokens were accepted or rejected.

The data flows in one direction:

```
Python instrumentation      JSON trace file       React timeline viewer
┌────────────────────┐     ┌───────────────┐     ┌──────────────────────┐
│ @profiled decorator │     │               │     │ Flame chart          │
│ TraceCollector      │────▸│ trace.json    │────▸│ Per-request swimlane │
│ context managers    │     │               │     │ Hover details        │
└────────────────────┘     └───────────────┘     └──────────────────────┘
```

---

## Architecture: Two Independent Halves

### Half 1: Python Instrumentation (`profiler/`)

A zero-dependency instrumentation library that records timestamped spans.
No React, no server, no frontend.
Produces a JSON file that can be analyzed with Python, loaded into Chrome's `chrome://tracing`, or fed to Half 2.

Key design constraints:
- **Must not change inference logic.** Instrumentation is additive - decorators and context managers, never modifying function signatures or return values.
- **Must support disabling.** A global flag (`PROFILING_ENABLED`) that makes all instrumentation no-ops when False. Zero overhead in production.
- **Must track per-request identity.** Every span is tagged with a `request_id` so the timeline can show per-request swimlanes.
- **Must be self-contained.** No imports from the inference files. The profiler is a tool, not a dependency.

### Half 2: React Timeline Viewer (`/profiler` route)

A new page in the existing frontend that loads a `trace.json` file and renders an interactive timeline.
Think Chrome DevTools Performance tab, but specialized for inference:

- **X-axis is wall-clock time** (milliseconds)
- **Y-axis is grouped by request** (swimlanes)
- **Spans are colored by category** (attention, sampling, scheduling, memory, prefill, decode)
- **Hover shows details** (duration, KV blocks allocated, tokens in batch, acceptance rate)
- **Zoom and pan** for large traces

---

## Half 1: Python Instrumentation

### The Span Data Model

```python
@dataclass
class Span:
    name: str              # e.g., "attention_forward", "kv_cache_assemble"
    category: str          # e.g., "attention", "memory", "scheduling", "sampling"
    start_us: int          # microsecond timestamp (time.perf_counter_ns() // 1000)
    end_us: int            # microsecond timestamp
    request_id: str | None # which request this belongs to (None = system-level)
    metadata: dict         # arbitrary key-value pairs for hover details
```

### The Trace Collector

```python
class TraceCollector:
    """Global singleton that accumulates spans during a profiled run."""

    def __init__(self):
        self.spans: list[Span] = []
        self.enabled: bool = False
        self._epoch_us: int = 0     # timestamp of first span (for relative times)

    def begin_trace(self):
        """Call once before the profiled run. Resets state."""
        self.spans = []
        self.enabled = True
        self._epoch_us = time.perf_counter_ns() // 1000

    def end_trace(self) -> list[Span]:
        """Call after the profiled run. Returns spans and disables collection."""
        self.enabled = False
        return self.spans

    def record(self, name, category, start_us, end_us, request_id=None, **metadata):
        if not self.enabled:
            return
        self.spans.append(Span(
            name=name,
            category=category,
            start_us=start_us - self._epoch_us,
            end_us=end_us - self._epoch_us,
            request_id=request_id,
            metadata=metadata,
        ))

    def export_json(self, path):
        """Write spans to a JSON file in Chrome Trace Event format."""
        ...
```

### Instrumentation Primitives

Three ways to instrument code, from least to most invasive:

**1. Context manager (for inline blocks):**

```python
with trace.span("kv_cache_assemble", category="memory", request_id=req.id):
    assembled = assemble_batch_cache(running_requests)
```

**2. Decorator (for functions):**

```python
@profiled("attention_forward", category="attention")
def forward(self, x, past_k=None, past_v=None, attn_mask=None):
    ...
```

**3. Manual record (for computed spans):**

```python
start = time.perf_counter_ns() // 1000
# ... do work ...
end = time.perf_counter_ns() // 1000
trace.record("spec_verify", "speculative", start, end, 
             request_id=req.id, 
             drafted=K, accepted=num_accepted)
```

### What to Instrument

The instrumentation targets map to the pipeline stages of your most complex implementations.
Start with `nanogpt-interleaving.py` (it has all the components) and work backward:

| Category | Spans to record | What it tells you |
|----------|----------------|-------------------|
| **scheduling** | `scheduler.schedule()`, `_maybe_admit()`, `_maybe_preempt()` | How long the scheduler takes per step. Is admission logic a bottleneck? |
| **memory** | `assemble_batch_cache()`, `disassemble_batch_cache()`, `maybe_allocate_block()`, `ensure_blocks_for_new_tokens()` | Time spent on KV cache assembly/disassembly. Is `torch.cat` padding slow? |
| **prefill** | The prefill chunk of `scheduled_generate()` - the forward pass on prefill tokens | How long each prefill chunk takes. Is chunking effective? |
| **attention** | `Head.forward()` or the batched attention call | Per-layer attention time. Dominated by matmul? |
| **sampling** | `F.softmax()` + `torch.multinomial()` | Sampling overhead per step. Usually tiny but worth confirming. |
| **speculative** | `draft_tokens()`, `verify_candidates()`, `accept_reject()`, `trim_kv_cache()` | Breakdown of spec decode: draft vs. verify vs. accept/reject time. |
| **cache_management** | `find_cached_prefix()`, `load_cached_blocks()`, hash computations | Prefix caching overhead. Is radix tree lookup fast enough? |
| **request_lifecycle** | Request admission, completion, preemption events | When each request enters/exits the running set. |

### Metadata Per Span

The metadata dict carries details that the timeline hover will display:

| Span | Metadata fields |
|------|----------------|
| `scheduler.schedule` | `batch_size`, `num_prefill`, `num_decode`, `token_budget_used` |
| `assemble_batch_cache` | `num_requests`, `max_cache_len`, `pad_tokens` |
| `prefill_chunk` | `request_id`, `chunk_start`, `chunk_end`, `chunk_tokens` |
| `attention_forward` | `layer`, `seq_len`, `batch_size` |
| `spec_draft` | `K`, `draft_tokens` |
| `spec_verify` | `K`, `num_accepted`, `acceptance_rate` |
| `block_allocate` | `request_id`, `block_id`, `total_allocated` |
| `request_admit` | `request_id`, `prompt_len`, `priority` |
| `request_complete` | `request_id`, `total_tokens`, `time_to_first_token` |

### Output Format: Chrome Trace Events

Use the Chrome Trace Event Format (the `chrome://tracing` JSON format).
This gives you two viewers for free: Chrome's built-in tracing viewer and Perfetto.
The React timeline is a third, specialized viewer.

```json
{
  "traceEvents": [
    {"name": "scheduler.schedule", "cat": "scheduling", "ph": "X", "ts": 0, "dur": 45, "pid": 0, "tid": "req_0", "args": {"batch_size": 3}},
    {"name": "assemble_batch_cache", "cat": "memory", "ph": "X", "ts": 45, "dur": 120, "pid": 0, "tid": "req_0", "args": {"num_requests": 3}},
    {"name": "attention_forward", "cat": "attention", "ph": "X", "ts": 165, "dur": 80, "pid": 0, "tid": "system", "args": {"layer": 0, "seq_len": 32}},
    ...
  ],
  "metadata": {
    "implementation": "nanogpt-interleaving",
    "model_params": 57000,
    "block_size": 64,
    "num_requests": 5,
    "total_steps": 20
  }
}
```

Using `tid` (thread ID) as the request ID gives per-request swimlanes in Chrome's viewer with zero extra work.

---

## Half 2: React Timeline Viewer

### New Files

```
frontend/src/
  ProfilerPage.jsx           # main page component
  ProfilerPage.css           # styles
  profiler/
    TimelineChart.jsx         # the flame-chart canvas component
    SpanTooltip.jsx           # hover tooltip with metadata
    SummaryPanel.jsx          # aggregate statistics sidebar
    CategoryLegend.jsx        # color legend for span categories
    useTraceData.js           # hook to load and process trace.json
```

### Route Addition

```jsx
// App.jsx
<Route path="/profiler" element={<ProfilerPage />} />
```

### The Timeline Chart

The core visualization is a `<canvas>` element (not SVG - too many spans for DOM nodes).
Each request is a horizontal swimlane.
Spans are rectangles positioned by `(start_us, end_us)` on the x-axis and by request on the y-axis.
Color encodes category.

```
Time (ms) →    0       5       10      15      20      25      30
              ├───────┼───────┼───────┼───────┼───────┼───────┤

req_0         ██████ prefill_chunk ████████  decode ██  decode ██  decode ██
                                      ↕
req_1                 ████ prefill ████████ decode ██ decode ██████ decode ████
                                      ↕
req_2                          ████████ prefill_chunk ████████  decode ██
                                      ↕
system        ▓▓ sched ▓▓     ▓▓ sched ▓▓      ▓▓ sched ▓▓     ▓▓ sched ▓▓
```

Within each request lane, spans can be stacked (nested spans become child rows):

```
req_0:   ████████████████ forward_pass ████████████████
         ██ attn_L0 ██ ██ attn_L1 ██ ██ attn_L2 ██ ██ attn_L3 ██ █ sample █
```

### Interaction

| Gesture | Behavior |
|---------|----------|
| Hover over span | Show tooltip with name, duration, metadata |
| Click span | Pin the tooltip, highlight all spans for the same request |
| Scroll wheel | Zoom in/out on the time axis |
| Click + drag | Pan the viewport |
| Click request label | Expand/collapse nested spans for that request |
| Double-click category in legend | Solo that category (hide all others) |

### Summary Panel

A sidebar that shows aggregate statistics computed from the trace:

```
┌─────────────────────────────────┐
│  Trace Summary                  │
│                                 │
│  Total time:     31.4 ms        │
│  Requests:       5              │
│  Total tokens:   127            │
│  Throughput:     4,045 tok/s    │
│                                 │
│  Time Breakdown                 │
│  ▓▓▓▓▓▓▓▓░░ Attention    62%   │
│  ▓▓░░░░░░░░ Memory       15%   │
│  ▓░░░░░░░░░ Scheduling    8%   │
│  ▓░░░░░░░░░ Sampling      5%   │
│  ▓░░░░░░░░░ Other        10%   │
│                                 │
│  Per-Request Latency            │
│  req_0: 28.3 ms (TTFT: 2.1ms)  │
│  req_1: 24.7 ms (TTFT: 5.3ms)  │
│  req_2: 19.1 ms (TTFT: 8.7ms)  │
│  req_3: 15.2 ms (TTFT: 12.4ms) │
│  req_4: 11.8 ms (TTFT: 15.1ms) │
│                                 │
│  Spec Decode (if applicable)    │
│  Acceptance rate: 42%           │
│  Avg accepted/round: 1.7       │
│  Verify overhead: 3.2 ms       │
└─────────────────────────────────┘
```

### File Loading

The page has a drag-and-drop zone (or file picker) to load a `trace.json`.
There's also a dropdown of pre-generated example traces bundled with the app:

- `trace_kv_cache.json` - simple KV cache, single request
- `trace_continuous_batching.json` - 5 requests with staggered arrivals
- `trace_chunked_prefill.json` - long prompt split into chunks
- `trace_paged_attention.json` - block allocation visible
- `trace_speculative.json` - draft/verify/accept cycle
- `trace_interleaving.json` - interleaved prefill+decode, the most complex

These example traces are generated by running the profiler on your existing implementations.

---

## The Trace Generation Scripts

### `profiler/generate_traces.py`

A script that imports each implementation, instruments it, runs a small workload, and writes the trace JSON:

```python
def generate_kv_cache_trace():
    """Instrument nanogpt-kv-cache.py and generate a single-request trace."""
    # 1. Train the model (or load a checkpoint)
    # 2. Create a TraceCollector
    # 3. Monkey-patch or wrap the generate function with instrumentation
    # 4. Run generation
    # 5. Export trace.json

def generate_continuous_batching_trace():
    """Instrument nanogpt-continuous-batching.py with 5 staggered requests."""
    # The interesting part: instrument scheduled_generate's inner loop
    # to record per-step scheduler decisions and batch assembly

def generate_speculative_trace():
    """Instrument nanogpt-spec-decode.py to show draft/verify/accept cycles."""
    # Record each draft_tokens() call, each verify_candidates() call,
    # and each accept_reject() decision with acceptance metadata
```

Each function produces a self-contained trace file.
The traces are committed to the repo under `profiler/traces/` so the frontend has example data without requiring a training run.

---

## Instrumenting Without Modifying Source Files

The key challenge: your inference files are self-contained scripts with top-level training code.
You can't `import nanogpt-interleaving` without triggering training.

Two approaches:

### Approach A: Monkey-patching (preferred for non-invasive profiling)

```python
import importlib.util

# Load the module without executing top-level code
spec = importlib.util.spec_from_file_location("engine", "nanogpt-interleaving.py")
mod = importlib.util.module_from_spec(spec)
# ... selectively exec only class/function definitions ...

# Wrap the functions we want to profile
original_schedule = mod.Scheduler.schedule
def profiled_schedule(self, step):
    with trace.span("scheduler.schedule", "scheduling", batch_size=len(self.running)):
        return original_schedule(self, step)
mod.Scheduler.schedule = profiled_schedule
```

### Approach B: Duplicate + instrument (preferred for clean traces)

Create a `profiler/profiled_engine.py` that copies the model/scheduler/generate code from `nanogpt-interleaving.py` (the most complex implementation) with instrumentation calls inlined.
This is the same pattern as `benchmarks/eval_runs.py` - self-contained duplication to avoid import side effects.

Approach B produces cleaner traces because you can instrument at exactly the granularity you want (e.g., inside a `for` loop body).
Approach A is less maintenance burden.

**Recommendation:** Start with Approach B for the initial implementation.
Once it works, evaluate whether Approach A is worth the complexity.

---

## Recommended Build Order

```
Phase 1: Instrumentation Core (Python only, no frontend)
  1. profiler/__init__.py              - Span dataclass, TraceCollector
  2. profiler/instrument.py            - @profiled decorator, span() context manager
  3. profiler/export.py                - Chrome Trace Event JSON export
  4. profiler/profiled_engine.py       - Instrumented copy of interleaving engine
  5. profiler/generate_traces.py       - Script to generate example traces
  6. Validate: open trace.json in chrome://tracing, verify it looks correct

Phase 2: React Timeline Core
  7. frontend/src/profiler/useTraceData.js      - Load + parse trace JSON
  8. frontend/src/profiler/CategoryLegend.jsx    - Color legend
  9. frontend/src/profiler/TimelineChart.jsx     - Canvas-based flame chart
  10. frontend/src/ProfilerPage.jsx              - Page shell with file loading
  11. frontend/src/ProfilerPage.css              - Styles
  12. Update App.jsx with /profiler route
  13. Validate: load an example trace, verify spans render correctly

Phase 3: Interactivity
  14. frontend/src/profiler/SpanTooltip.jsx      - Hover tooltip
  15. Add zoom/pan to TimelineChart
  16. Add click-to-pin and request highlighting
  17. frontend/src/profiler/SummaryPanel.jsx      - Aggregate stats sidebar
  18. Validate: verify hover, zoom, pan, summary all work

Phase 4: Multiple Implementation Traces
  19. Add KV cache trace generation
  20. Add continuous batching trace generation
  21. Add speculative decoding trace generation
  22. Add paged attention trace generation
  23. Bundle traces as example data in frontend
  24. Add dropdown to select pre-generated traces

Phase 5: Polish
  25. Add expand/collapse for nested spans per request
  26. Add category filtering (double-click legend to solo)
  27. Add time-to-first-token markers on request lanes
  28. Add the profiler to the landing page navigation
  29. Update README with profiler section
```

---

## Open Questions

**Q1: Canvas vs. SVG for the timeline chart?**
Canvas is better for large traces (1000+ spans).
SVG is easier for interactivity (click/hover is built-in).
For traces from NanoGPT (typically 100-500 spans), SVG might be fine.
For scaling to real engines, canvas is necessary.
Recommendation: start with SVG, migrate to canvas if performance is an issue.

**Q2: How granular should attention instrumentation be?**
Options:
- Per-layer (4 spans per step for a 4-layer model) - useful for seeing layer balance
- Per-head (16 spans per step for 4 layers x 4 heads) - reveals head-level patterns but very noisy
- Per-forward-pass (1 span per step) - cleanest, least informative
Recommendation: start with per-layer.
Per-head can be added later behind a verbosity flag.

**Q3: Should the profiler run as a live server or a batch tool?**
Options:
- Batch: run profiler script, produce trace.json, load in frontend
- Live: WebSocket from Python to React, spans stream in real-time
Recommendation: batch first.
Live streaming is much more complex (buffering, backpressure, partial renders) and the educational value is the same.
The simulation page already has step-by-step animation - the profiler can be static and still be valuable.

**Q4: How to handle the training phase?**
The profiler should skip training and only instrument inference.
Either load a pre-trained checkpoint or train at the start of `generate_traces.py` (like `eval_runs.py` does) and exclude training spans from the trace.

---

## File Structure

```
profiler/
  __init__.py                 # Span, TraceCollector, global instance
  instrument.py               # @profiled decorator, span() context manager
  export.py                   # Chrome Trace Event JSON serializer
  profiled_engine.py          # Instrumented copy of interleaving engine
  generate_traces.py          # Script to produce example trace files
  traces/                     # Pre-generated example traces
    trace_kv_cache.json
    trace_continuous_batching.json
    trace_chunked_prefill.json
    trace_paged_attention.json
    trace_speculative.json
    trace_interleaving.json

frontend/src/
  ProfilerPage.jsx            # Main profiler page
  ProfilerPage.css            # Profiler styles
  profiler/
    TimelineChart.jsx          # Flame chart component
    SpanTooltip.jsx            # Hover tooltip
    SummaryPanel.jsx           # Aggregate statistics
    CategoryLegend.jsx         # Color legend
    useTraceData.js            # Trace data loader hook
```

---

## What Makes This Useful Beyond Education

Most inference profiling tools are either:
- **Too low-level:** `torch.profiler` / CUDA profiling shows kernel-level detail but not request-level semantics.
You see "gemm" and "softmax" but not "this is the prefill chunk for request 3."
- **Too high-level:** Request latency dashboards show P50/P99 but not *why* a request was slow.
Was it waiting for a batch slot? Was it preempted? Was speculative decoding less effective?

This profiler sits in the middle: it tracks semantic operations (scheduler decisions, prefill chunks, spec decode rounds) with wall-clock timing.
The per-request swimlane view makes it immediately obvious when a request is stalled (gap in its lane) and why (another request's prefill is monopolizing the GPU).

Production inference teams build exactly this kind of tooling internally.
Open-sourcing it alongside the from-scratch implementations would be genuinely novel.
