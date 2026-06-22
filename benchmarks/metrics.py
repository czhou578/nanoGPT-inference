"""
Lightweight inference metrics collector for NanoGPT.

Thread-safe counters, gauges, and histograms with no external dependencies.
Supports JSON snapshot and Prometheus exposition format.

Usage:
    from benchmarks.metrics import InferenceMetrics

    metrics = InferenceMetrics()
    metrics.inc("requests_total")
    metrics.set_gauge("active_requests", 3)
    metrics.observe("ttft_seconds", 0.042)

    print(metrics.snapshot())       # JSON-serializable dict
    print(metrics.prometheus_text())  # Prometheus exposition format
"""

import threading
import time
from dataclasses import dataclass
from statistics import mean, stdev


# ──────────────────────────────────────────────────────────────────────
# PercentileStats (reused from load_tester for consistency)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HistogramSummary:
    """Summary of a histogram's current distribution."""
    count: int
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "p50": round(self.p50, 6),
            "p90": round(self.p90, 6),
            "p95": round(self.p95, 6),
            "p99": round(self.p99, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
        }


def _compute_histogram_summary(values: list[float]) -> HistogramSummary:
    """Compute percentile summary from raw values."""
    if not values:
        return HistogramSummary(
            count=0, p50=0, p90=0, p95=0, p99=0,
            min=0, max=0, mean=0, std=0,
        )
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return HistogramSummary(
        count=n,
        p50=sorted_vals[int(n * 0.50)],
        p90=sorted_vals[min(int(n * 0.90), n - 1)],
        p95=sorted_vals[min(int(n * 0.95), n - 1)],
        p99=sorted_vals[min(int(n * 0.99), n - 1)],
        min=sorted_vals[0],
        max=sorted_vals[-1],
        mean=mean(sorted_vals),
        std=stdev(sorted_vals) if n > 1 else 0.0,
    )


# ──────────────────────────────────────────────────────────────────────
# InferenceMetrics
# ──────────────────────────────────────────────────────────────────────

class InferenceMetrics:
    """
    Thread-safe metrics collector with counters, gauges, and histograms.

    Counters: monotonically increasing values (e.g., total requests).
    Gauges: point-in-time values that go up and down (e.g., active requests).
    Histograms: append-only lists of observations for percentile analysis.
    """

    def __init__(self, max_histogram_size: int = 10000):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._max_histogram_size = max_histogram_size
        self._start_time = time.monotonic()

    # ── Counters ──────────────────────────────────────────────────────

    def inc(self, name: str, value: int = 1):
        """Increment a counter by value (default 1)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    # ── Gauges ────────────────────────────────────────────────────────

    def set_gauge(self, name: str, value: float):
        """Set a gauge to an absolute value."""
        with self._lock:
            self._gauges[name] = value

    def inc_gauge(self, name: str, value: float = 1.0):
        """Increment a gauge (can be negative to decrement)."""
        with self._lock:
            self._gauges[name] = self._gauges.get(name, 0.0) + value

    def get_gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    # ── Histograms ────────────────────────────────────────────────────

    def observe(self, name: str, value: float):
        """Record a single observation in a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            hist = self._histograms[name]
            hist.append(value)
            # Evict oldest entries if over limit
            if len(hist) > self._max_histogram_size:
                self._histograms[name] = hist[-self._max_histogram_size:]

    def get_histogram(self, name: str) -> HistogramSummary:
        with self._lock:
            values = list(self._histograms.get(name, []))
        return _compute_histogram_summary(values)

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return all metrics as a JSON-serializable dict."""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histogram_names = list(self._histograms.keys())
            histogram_values = {k: list(v) for k, v in self._histograms.items()}

        histograms = {}
        for name in histogram_names:
            histograms[name] = _compute_histogram_summary(
                histogram_values[name]
            ).to_dict()

        return {
            "uptime_seconds": round(time.monotonic() - self._start_time, 2),
            "counters": counters,
            "gauges": {k: round(v, 4) for k, v in gauges.items()},
            "histograms": histograms,
        }

    # ── Prometheus exposition format ──────────────────────────────────

    def prometheus_text(self) -> str:
        """
        Format metrics in Prometheus text exposition format.

        Counters → TYPE counter
        Gauges → TYPE gauge
        Histograms → summary-style quantile lines
        """
        lines = []

        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histogram_names = list(self._histograms.keys())
            histogram_values = {k: list(v) for k, v in self._histograms.items()}

        # Counters
        for name, value in sorted(counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in sorted(gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value:.4f}")

        # Histograms (as summary-style quantiles)
        for name in sorted(histogram_names):
            summary = _compute_histogram_summary(histogram_values[name])
            lines.append(f"# TYPE {name} summary")
            lines.append(f'{name}{{quantile="0.5"}} {summary.p50:.6f}')
            lines.append(f'{name}{{quantile="0.9"}} {summary.p90:.6f}')
            lines.append(f'{name}{{quantile="0.95"}} {summary.p95:.6f}')
            lines.append(f'{name}{{quantile="0.99"}} {summary.p99:.6f}')
            lines.append(f"{name}_count {summary.count}")
            lines.append(f"{name}_min {summary.min:.6f}")
            lines.append(f"{name}_max {summary.max:.6f}")
            lines.append(f"{name}_mean {summary.mean:.6f}")

        return "\n".join(lines) + "\n"

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self):
        """Clear all metrics. Useful for testing."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.monotonic()
