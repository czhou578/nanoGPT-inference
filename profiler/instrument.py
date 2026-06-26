"""
Instrumentation primitives: context manager, decorator, and event markers.

These are the three ways to add profiling spans to inference code:

1. Context manager (for inline blocks):
   with trace_span("kv_assemble", "memory", request_id="req_0"):
       assembled = assemble_batch_cache(reqs)

2. Decorator (for functions):
   @profiled("attention_forward", "attention")
   def forward(self, x, past_k=None, past_v=None):
       ...

3. Event marker (for instantaneous events):
   trace_event("request_admitted", "lifecycle", request_id="req_0", prompt_len=16)
"""

import functools
from contextlib import contextmanager

from profiler import trace


@contextmanager
def trace_span(name: str, category: str, request_id: str | None = None, **metadata):
    """
    Context manager that records a span for the duration of the block.

    Usage:
        with trace_span("assemble_batch_cache", "memory", num_requests=3):
            result = assemble_batch_cache(requests)

    When tracing is disabled, this is a trivial no-op (just yields).
    """
    if not trace.enabled:
        yield
        return

    start = trace._now()
    try:
        yield
    finally:
        end = trace._now()
        trace.record(name, category, start, end, request_id=request_id, **metadata)


def profiled(name: str, category: str, request_id_arg: str | None = None):
    """
    Decorator that wraps a function in a trace span.

    Args:
        name:            span name (e.g., "attention_forward")
        category:        span category (e.g., "attention")
        request_id_arg:  if set, the name of a keyword argument on the decorated
                         function whose value is the request_id. For example,
                         @profiled("prefill_chunk", "prefill", request_id_arg="request_id")
                         will extract request_id from the function's kwargs.

    Usage:
        @profiled("model_forward", "compute")
        def forward(self, idx, targets=None, pos=None, past_kvs=None):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not trace.enabled:
                return fn(*args, **kwargs)

            req_id = None
            if request_id_arg is not None:
                req_id = kwargs.get(request_id_arg)

            start = trace._now()
            try:
                return fn(*args, **kwargs)
            finally:
                end = trace._now()
                trace.record(name, category, start, end, request_id=req_id)

        return wrapper
    return decorator


def trace_event(name: str, category: str, request_id: str | None = None, **metadata):
    """
    Record an instantaneous event (zero-duration span).

    Use for state transitions: request admitted, request completed,
    preemption events, etc.

    Usage:
        trace_event("request_admitted", "lifecycle",
                    request_id="req_0", prompt_len=16, priority=1)
    """
    if not trace.enabled:
        return
    now = trace._now()
    trace.record(name, category, now, now, request_id=request_id, **metadata)
