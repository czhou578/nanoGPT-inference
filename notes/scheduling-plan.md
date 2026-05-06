# Scheduling Policies — Implementation Plan & Hints

## The Problem You're Solving

Your current continuous batching engine uses a naive FCFS (First-Come, First-Served) policy: requests are processed in arrival order. Prefilling requests block later arrivals from being admitted until they're done.

But imagine a real server:
- A very long, low-priority batch job arrives first and hogs the token budget for many steps
- A short, high-priority interactive request arrives a step later and waits behind it

A proper scheduler can **preempt** the low-priority job, serve the high-priority one immediately, and resume the evicted request when resources free up. This is exactly how vLLM's scheduler manages competing requests under memory pressure.

---

## Hint 1: Clean Up the Request Dataclass

Before building scheduling logic, add two fields to `Request`:

- `priority: int` — lower number = higher priority (0 is highest). Can be set per-request at creation time.
- `arrival_time: int` — the simulation step at which this request arrived. Used as the tiebreaker for FCFS within the same priority tier.

```python
@dataclass
class Request:
    ...
    priority: int = 0         # 0 = highest priority
    arrival_time: int = 0     # set when admitted to the scheduler
```

Your queue currently uses a list that you iterate with `queue_idx`. Think about what data structure would make ordering by `(priority, arrival_time)` efficient. A **heap** (`heapq`) is perfect here — pushing with a `(priority, arrival_time, req)` tuple gives you a priority queue for free.

**Question to ask yourself:** When two requests have the same priority, which should be served first — the one that arrived earlier or the one with fewer tokens left?

---

## Hint 2: Build a Proper Scheduler Class

Instead of inline scheduler logic inside `generate()`, extract it into a `Scheduler` class. This keeps the generate loop clean and makes it easy to swap policies.

```python
class Scheduler:
    def __init__(self, policy="fcfs", max_batch_size=4, token_budget=16, max_kv_tokens=256):
        self.policy = policy          # "fcfs" or "priority"
        self.max_batch_size = max_batch_size
        self.token_budget = token_budget
        self.max_kv_tokens = max_kv_tokens  # total KV slots available (memory budget)

        self.waiting = []             # heap of (sort_key, req) for incoming requests
        self.prefilling = []          # at most 1 request being chunked
        self.active = []              # requests in decode
        self.preempted = []           # evicted requests (cache cleared, must re-prefill)
```

The `schedule()` method runs once per step and returns which requests are doing what this step. The generate loop just calls `scheduler.schedule()` and acts on the result.

---

## Hint 3: FCFS is Just Priority = Arrival Time

Your two policies (FCFS and priority) can share the exact same heap — the only difference is the sort key:

- **FCFS:** sort key = `(0, arrival_time)` — all requests have equal priority, ordered by when they arrived.
- **Priority:** sort key = `(priority_score, arrival_time)` — lower priority score goes first; arrival time breaks ties.

```python
def _sort_key(self, req):
    if self.policy == "fcfs":
        return (0, req.arrival_time)
    elif self.policy == "priority":
        return (req.priority, req.arrival_time)
```

When a new request arrives, you push `(self._sort_key(req), req)` onto the heap. `heapq.heappop()` always gives you the highest-priority request to admit next.

---

## Hint 4: Memory Budgeting — When to Preempt

This is the core of real scheduling. You have a finite `max_kv_tokens` — a cap on how many total KV cache tokens all active requests can hold simultaneously.

At the start of each step, compute:
```
total_kv_used = sum of (len(prompt_tokens) + num_generated) for all active + prefilling requests
```

If admitting a new prefilling request would push `total_kv_used` over `max_kv_tokens`, you can't admit it yet.

**The preemption path:** If the system is over budget (can happen if a request generates more tokens than expected), you need to **evict** one or more active requests:

1. Pick the lowest-priority active request (the one you'd sacrifice first).
2. Call `req.clear_cache()` to free its KV memory.
3. Reset `req.prefill_cursor = 0` and `req.status = "waiting"`.
4. Move it to `self.preempted` (a separate list so you don't lose it).

When memory frees up again, the preempted request re-enters the waiting queue and must re-prefill from scratch. This is the **recompute preemption** strategy — as opposed to vLLM's **swap** strategy (offloading KV to CPU RAM instead of discarding it).

**Question to ask yourself:** Should preempted requests go to the front of the waiting queue (preserving their original priority) or to the back?

---

## Hint 5: The Scheduler's `schedule()` Method

Each step, the scheduler should return a structured decision. Here's a sketch:

```python
def schedule(self, step: int):
    """
    Returns:
        prefill_req:  Request | None   — the request getting a prefill chunk this step
        decode_reqs:  List[Request]    — requests getting a decode token this step
    """
    self._maybe_admit(step)       # promote waiting → prefilling if memory allows
    self._maybe_preempt()         # evict if over memory budget

    prefill_req = self.prefilling[0] if self.prefilling else None
    decode_reqs = list(self.active)

    return prefill_req, decode_reqs
```

`_maybe_admit()` checks if memory + batch size allow promoting the top of the waiting heap into `self.prefilling`. `_maybe_preempt()` checks if the current KV usage is over budget and evicts accordingly.

---

## Hint 6: Wiring the Scheduler into the Generate Loop

Your `scheduled_generate` function becomes much cleaner:

```python
def scheduled_generate(model, requests, policy="fcfs", token_budget=16, max_kv_tokens=256):
    scheduler = Scheduler(policy=policy, token_budget=token_budget, max_kv_tokens=max_kv_tokens)

    for req in requests:
        scheduler.add_request(req)

    while not scheduler.is_done():
        prefill_req, decode_reqs = scheduler.schedule(step)

        # --- Prefill chunk (same logic as chunked-prefill notebook) ---
        if prefill_req:
            ...run prefill chunk...
            if prefill_req.is_fully_prefilled:
                scheduler.promote(prefill_req)  # prefilling → active

        # --- Batched decode (same as before) ---
        if decode_reqs:
            ...run decode batch...
            for req in decode_reqs:
                if req.is_done:
                    scheduler.complete(req)      # active → done
```

The generate loop doesn't need to know about policies or memory — that's all encapsulated in `Scheduler`.

---

## Hint 7: Observing the Difference Between Policies

To make the difference between FCFS and Priority actually visible, set up a scenario where they diverge:

```python
requests = [
    Request(id=0, prompt_tokens=encode("A " * 30), max_new_tokens=20, priority=2),  # low priority, long
    Request(id=1, prompt_tokens=encode("B " * 5),  max_new_tokens=5,  priority=0),  # high priority, short
]
```

With FCFS, req 0 gets admitted first and hogs the budget. Req 1 waits.
With Priority, req 1 jumps the queue and completes immediately, then req 0 runs uncontested.

Print a per-step log showing which requests are active, which are waiting, and which are preempted. This is your visual proof that the scheduler is working correctly.

---

## Hint 8: Preemption in Practice

Preemption only triggers when `max_kv_tokens` is small enough that you can't hold all active requests in memory simultaneously. Design your test to make this happen:

```python
# Each token uses (n_layer * n_head * head_size * 2) floats of KV cache.
# For nanoGPT (4 layers, 4 heads, 16 head_size): 4 * 4 * 16 * 2 = 512 floats per token.
# Set max_kv_tokens = 64 → only ~64 total sequence positions across all requests.
```

When preemption fires, you should see a log line like:
```
[step 7] ⚠️  Preempting req 0 (priority=2, kv_tokens=45) to free memory for req 1
[step 8] Re-admitting req 0 — must re-prefill from scratch
```

---

## Test Scenarios

### Test 1: FCFS correctness
All requests same priority. Verify they complete in arrival order and output matches chunked-prefill notebook for the same prompts.

### Test 2: Priority jumping the queue
One low-priority long request + one high-priority short request. Verify the short one completes first under Priority policy, but second under FCFS.

### Test 3: Preemption under memory pressure
Set `max_kv_tokens` low enough to force at least one preemption. Verify:
- The evicted request's cache is cleared.
- It re-prefills correctly and produces valid output.
- Total output tokens match what you'd get without preemption (just slower).

### Test 4: Preempted request re-enters correctly
After preemption, the re-admitted request should produce **identical output** to if it had never been preempted (same random seed). This validates that your `clear_cache` + full re-prefill is correct.

---

## Summary of New Components vs Chunked Prefill

| Component | What's New |
|-----------|-----------|
| `Request` dataclass | Add `priority`, `arrival_time` fields |
| `Scheduler` class | New — encapsulates waiting heap, preemption logic, memory budgeting |
| `scheduled_generate` | Refactor of `continuous_batching_generate` — calls `scheduler.schedule()` each step |
| Model / Head / assemble_batch_cache | **Nothing changes** — scheduling is pure Python above the model |

The key insight: **the model doesn't know anything about scheduling**. All of this complexity lives in the 50–100 lines of `Scheduler` Python above it.
