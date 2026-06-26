"""
Inference Profiler - Core data model and trace collector.

The profiler records timestamped spans during inference execution.
Each span captures a named operation (e.g., "attention_forward",
"assemble_batch_cache") with its category, duration, request association,
and arbitrary metadata.

Usage:
    from profiler import trace

    trace.begin_trace()
    # ... run inference ...
    trace.end_trace()
    trace.export_json("trace.json")

The trace collector is a global singleton. Instrumentation calls are
no-ops when tracing is disabled, adding zero overhead.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Span:
    """A single profiled operation."""
    name: str                           # e.g., "attention_forward", "scheduler.schedule"
    category: str                       # e.g., "attention", "scheduling", "memory"
    start_us: int                       # microsecond timestamp (relative to trace epoch)
    end_us: int                         # microsecond timestamp (relative to trace epoch)
    request_id: Optional[str] = None    # which request this belongs to (None = system-level)
    metadata: dict = field(default_factory=dict)

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


class TraceCollector:
    """
    Global singleton that accumulates spans during a profiled run.

    All timestamps are stored relative to the trace epoch (the moment
    begin_trace() was called), so the first span always starts near t=0.
    """

    def __init__(self):
        self.spans: list[Span] = []
        self.enabled: bool = False
        self._epoch_us: int = 0

    def begin_trace(self):
        """Call once before the profiled run. Resets state and starts the clock."""
        self.spans = []
        self.enabled = True
        self._epoch_us = time.perf_counter_ns() // 1000

    def end_trace(self) -> list[Span]:
        """Call after the profiled run. Returns collected spans and disables tracing."""
        self.enabled = False
        return self.spans

    def _now(self) -> int:
        """Current time in microseconds, relative to trace epoch."""
        return (time.perf_counter_ns() // 1000) - self._epoch_us

    def record(self, name: str, category: str, start_us: int, end_us: int,
               request_id: Optional[str] = None, **metadata):
        """
        Record a completed span.

        Args:
            name:       operation name
            category:   one of: attention, memory, scheduling, sampling,
                        prefill, decode, speculative, cache_management, lifecycle
            start_us:   start time (from _now())
            end_us:     end time (from _now())
            request_id: associated request ID, or None for system-level spans
            **metadata: arbitrary key-value pairs for the timeline tooltip
        """
        if not self.enabled:
            return
        self.spans.append(Span(
            name=name,
            category=category,
            start_us=start_us,
            end_us=end_us,
            request_id=request_id,
            metadata=metadata,
        ))

    def export_json(self, path: str, **extra_metadata):
        """
        Write spans to a JSON file in Chrome Trace Event Format.

        The output can be loaded in:
          - chrome://tracing
          - Perfetto (https://ui.perfetto.dev)
          - The React timeline viewer (/profiler)

        Args:
            path:            output file path
            **extra_metadata: top-level metadata fields (implementation name, etc.)
        """
        from profiler.export import export_chrome_trace
        export_chrome_trace(self.spans, path, **extra_metadata)

    def summary(self) -> dict:
        """
        Compute aggregate statistics from collected spans.

        Returns a dict with per-category total time, span counts, and
        per-request latency breakdown.
        """
        if not self.spans:
            return {}

        by_category: dict[str, list[Span]] = {}
        by_request: dict[str, list[Span]] = {}

        for span in self.spans:
            by_category.setdefault(span.category, []).append(span)
            if span.request_id is not None:
                by_request.setdefault(span.request_id, []).append(span)

        total_us = max(s.end_us for s in self.spans) - min(s.start_us for s in self.spans)

        category_summary = {}
        for cat, spans in sorted(by_category.items()):
            cat_total = sum(s.duration_us for s in spans)
            category_summary[cat] = {
                "total_us": cat_total,
                "count": len(spans),
                "pct": round(100 * cat_total / total_us, 1) if total_us > 0 else 0,
            }

        request_summary = {}
        for req_id, spans in sorted(by_request.items()):
            req_start = min(s.start_us for s in spans)
            req_end = max(s.end_us for s in spans)
            request_summary[req_id] = {
                "total_us": req_end - req_start,
                "span_count": len(spans),
            }

        return {
            "total_us": total_us,
            "total_spans": len(self.spans),
            "by_category": category_summary,
            "by_request": request_summary,
        }


# Global singleton - import this from anywhere
trace = TraceCollector()
