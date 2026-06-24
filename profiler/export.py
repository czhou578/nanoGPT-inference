"""
Export profiler spans to Chrome Trace Event Format (JSON).

The Chrome Trace Event Format is a well-documented JSON schema supported by:
  - chrome://tracing (built into Chrome)
  - Perfetto (https://ui.perfetto.dev)
  - Custom viewers (our React /profiler page)

Format spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU

We use "X" (complete) events for spans and "i" (instant) events for markers.
Each request gets its own thread ID (tid) so Chrome's viewer automatically
creates per-request swimlanes.
"""

import json
from dataclasses import asdict
from profiler import Span


def export_chrome_trace(spans: list[Span], path: str, **extra_metadata):
    """
    Write spans to a JSON file in Chrome Trace Event Format.

    Args:
        spans:            list of Span objects from TraceCollector
        path:             output file path
        **extra_metadata: fields added to the top-level metadata
                          (e.g., implementation="interleaving", num_requests=5)
    """
    events = []

    for span in spans:
        duration = span.end_us - span.start_us

        # Use "tid" as the request lane. System-level spans go to "system".
        tid = span.request_id if span.request_id is not None else "system"

        if duration == 0:
            # Instant event (trace_event calls)
            event = {
                "name": span.name,
                "cat": span.category,
                "ph": "i",           # instant event
                "ts": span.start_us,
                "pid": 0,
                "tid": tid,
                "s": "t",            # scope = thread
                "args": span.metadata,
            }
        else:
            # Complete event (trace_span / @profiled calls)
            event = {
                "name": span.name,
                "cat": span.category,
                "ph": "X",           # complete event
                "ts": span.start_us,
                "dur": duration,
                "pid": 0,
                "tid": tid,
                "args": span.metadata,
            }

        events.append(event)

    # Build the output object
    output = {
        "traceEvents": events,
        "metadata": extra_metadata,
    }

    # Also embed a summary for the React viewer
    if spans:
        by_category: dict[str, int] = {}
        by_request: dict[str, dict] = {}
        total_us = max(s.end_us for s in spans) - min(s.start_us for s in spans)

        for span in spans:
            dur = span.end_us - span.start_us
            by_category[span.category] = by_category.get(span.category, 0) + dur

            if span.request_id is not None:
                if span.request_id not in by_request:
                    by_request[span.request_id] = {"start_us": span.start_us, "end_us": span.end_us}
                else:
                    entry = by_request[span.request_id]
                    entry["start_us"] = min(entry["start_us"], span.start_us)
                    entry["end_us"] = max(entry["end_us"], span.end_us)

        output["summary"] = {
            "total_us": total_us,
            "total_spans": len(spans),
            "by_category": {cat: {"total_us": t, "pct": round(100 * t / total_us, 1)}
                            for cat, t in sorted(by_category.items())},
            "by_request": {req: {"total_us": d["end_us"] - d["start_us"]}
                           for req, d in sorted(by_request.items())},
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Trace written to {path} ({len(events)} events, {len(spans)} spans)")
