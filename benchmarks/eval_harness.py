"""
Eval harness for NanoGPT inference engine — quality regression detection.

Measures output quality metrics that are ORTHOGONAL to throughput benchmarks:
  - Perplexity on held-out validation data
  - Repetition ratio in generated text
  - N-gram diversity (distinct-2, distinct-3)
  - Self-BLEU (intra-generation diversity)
  - Deterministic consistency (same seed → same output)

All metrics run on CPU. No GPU required.

Usage:
    from benchmarks.eval_harness import EvalHarness

    harness = EvalHarness(model, val_data, vocab_size, device="cpu", block_size=64)
    result = harness.run_full_eval(generate_fn, "kv_cache")
    report = harness.compare_to_baseline(result, baseline_result)

The generate_fn signature must be:
    def generate_fn(model, idx, max_new_tokens) -> torch.Tensor
    where idx is (1, T) and the return is (1, T + max_new_tokens)
"""

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Quality metrics for one implementation."""
    implementation: str
    perplexity: float
    repetition_ratio: float       # 0 = all unique, 1 = all repeated
    distinct_2: float             # distinct bigrams / total bigrams
    distinct_3: float             # distinct trigrams / total trigrams
    self_bleu: float              # avg BLEU between pairs of generations
    consistency: float            # 1.0 = perfectly reproducible across seeds
    num_prompts: int
    num_tokens_generated: int
    eval_time_seconds: float
    timestamp: str                # ISO 8601
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(**d)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "EvalResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class RegressionFlag:
    """A single regression signal."""
    metric: str
    baseline_value: float
    current_value: float
    threshold_pct: float
    delta_pct: float
    is_regression: bool
    direction: str   # "higher_is_worse" or "lower_is_worse"


@dataclass
class RegressionReport:
    """Comparison of current eval against a baseline."""
    implementation: str
    baseline_implementation: str
    flags: list  # list of RegressionFlag
    overall_pass: bool
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "implementation": self.implementation,
            "baseline_implementation": self.baseline_implementation,
            "flags": [asdict(f) for f in self.flags],
            "overall_pass": self.overall_pass,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────
# Quality metric functions (stateless, pure)
# ──────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    val_data: torch.Tensor,
    *,
    device: str,
    block_size: int,
    num_windows: int = 50,
    window_size: int = 32,
    model_returns_3: bool = False,
) -> float:
    """
    Sliding-window perplexity on validation data.

    Picks `num_windows` random windows of `window_size` tokens from val_data,
    runs a forward pass, and computes the average cross-entropy loss.
    Perplexity = exp(avg_loss).

    Args:
        model_returns_3: If True, model returns (logits, loss, kvs).
                         If False, model returns (logits, loss).
    """
    model.eval()
    total_loss = 0.0
    count = 0

    torch.manual_seed(42)
    max_start = len(val_data) - window_size - 1
    if max_start <= 0:
        return float("inf")

    with torch.no_grad():
        for _ in range(num_windows):
            start = torch.randint(0, max_start, (1,)).item()
            x = val_data[start : start + window_size].unsqueeze(0).to(device)
            y = val_data[start + 1 : start + window_size + 1].unsqueeze(0).to(device)

            result = model(x, y)
            if model_returns_3:
                _, loss, _ = result
            else:
                _, loss = result

            if loss is not None:
                total_loss += loss.item()
                count += 1

    if count == 0:
        return float("inf")

    avg_loss = total_loss / count
    return math.exp(avg_loss)


def compute_repetition_ratio(
    tokens: list[int],
    window_size: int = 20,
) -> float:
    """
    Fraction of tokens that are repeated within a sliding window.

    Returns a value between 0 (all unique within every window) and 1 (all repeated).
    This catches degenerate repetition loops.
    """
    if len(tokens) < window_size:
        if len(tokens) == 0:
            return 0.0
        unique = len(set(tokens))
        return 1.0 - (unique / len(tokens))

    total_repetition = 0.0
    num_windows = 0

    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        unique = len(set(window))
        repetition = 1.0 - (unique / window_size)
        total_repetition += repetition
        num_windows += 1

    return total_repetition / num_windows if num_windows > 0 else 0.0


def compute_distinct_n(tokens: list[int], n: int) -> float:
    """
    Distinct-N: the ratio of unique n-grams to total n-grams.

    Higher = more diverse output. A value of 1.0 means every n-gram is unique.
    This is a standard text diversity metric from Li et al. (2016).
    """
    if len(tokens) < n:
        return 1.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0

    return len(set(ngrams)) / len(ngrams)


def compute_self_bleu(
    generations: list[list[int]],
    max_n: int = 4,
) -> float:
    """
    Self-BLEU: average BLEU between every pair of generations.

    Lower self-BLEU = more diverse outputs across different prompts/seeds.
    Uses a simplified BLEU-N implementation (no brevity penalty).
    """
    if len(generations) < 2:
        return 0.0

    def _ngram_counts(tokens, n):
        counts = Counter()
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i : i + n])] += 1
        return counts

    def _bleu_pair(hypothesis: list[int], reference: list[int]) -> float:
        """Simplified BLEU between one hypothesis and one reference."""
        if not hypothesis or not reference:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            if len(hypothesis) < n:
                precisions.append(0.0)
                continue

            hyp_counts = _ngram_counts(hypothesis, n)
            ref_counts = _ngram_counts(reference, n)

            clipped = sum(
                min(count, ref_counts.get(ngram, 0))
                for ngram, count in hyp_counts.items()
            )
            total = sum(hyp_counts.values())

            precisions.append(clipped / total if total > 0 else 0.0)

        # Geometric mean of precisions (avoid log(0))
        if any(p == 0 for p in precisions):
            return 0.0
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        return math.exp(log_avg)

    scores = []
    for i in range(len(generations)):
        for j in range(i + 1, len(generations)):
            scores.append(_bleu_pair(generations[i], generations[j]))

    return sum(scores) / len(scores) if scores else 0.0


def compute_consistency(
    generate_fn: Callable,
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    device: str,
    num_trials: int = 3,
    seed: int = 42,
) -> float:
    """
    Deterministic consistency: generate with the same seed multiple times.

    Returns 1.0 if all trials produce identical output, 0.0 if none match.
    This catches non-determinism bugs (race conditions, uninitialized memory).
    """
    outputs = []

    for _ in range(num_trials):
        torch.manual_seed(seed)
        with torch.no_grad():
            result = generate_fn(model, prompt.clone().to(device), max_new_tokens)
            if isinstance(result, torch.Tensor):
                tokens = result[0].tolist()
            else:
                tokens = result
        outputs.append(tokens)

    if not outputs:
        return 1.0

    reference = outputs[0]
    matches = sum(1 for out in outputs[1:] if out == reference)
    return (matches + 1) / len(outputs)  # +1 for the reference matching itself


# ──────────────────────────────────────────────────────────────────────
# EvalHarness — the main class
# ──────────────────────────────────────────────────────────────────────

class EvalHarness:
    """
    Quality evaluation harness for NanoGPT implementations.

    Runs a battery of quality tests on CPU and produces an EvalResult
    that can be compared against a frozen baseline for regression detection.
    """

    def __init__(
        self,
        model,
        val_data: torch.Tensor,
        vocab_size: int,
        *,
        device: str = "cpu",
        block_size: int = 64,
        model_returns_3: bool = False,
    ):
        self.model = model
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.device = device
        self.block_size = block_size
        self.model_returns_3 = model_returns_3

    def _make_prompts(
        self,
        num_prompts: int = 20,
        prompt_len: int = 16,
        seed: int = 42,
    ) -> list[torch.Tensor]:
        """Generate deterministic prompts from validation data."""
        prompts = []
        torch.manual_seed(seed)

        max_start = len(self.val_data) - prompt_len
        if max_start <= 0:
            # Fallback: use random tokens
            for _ in range(num_prompts):
                p = torch.randint(0, self.vocab_size, (1, prompt_len))
                prompts.append(p)
            return prompts

        for _ in range(num_prompts):
            start = torch.randint(0, max_start, (1,)).item()
            p = self.val_data[start : start + prompt_len].unsqueeze(0)
            prompts.append(p)

        return prompts

    def eval_perplexity(
        self,
        num_windows: int = 50,
        window_size: int = 32,
    ) -> float:
        """Compute perplexity on validation data."""
        return compute_perplexity(
            self.model,
            self.val_data,
            device=self.device,
            block_size=self.block_size,
            num_windows=num_windows,
            window_size=min(window_size, self.block_size),
            model_returns_3=self.model_returns_3,
        )

    def eval_generation_quality(
        self,
        generate_fn: Callable,
        *,
        num_prompts: int = 20,
        prompt_len: int = 16,
        max_new_tokens: int = 50,
        num_seeds: int = 3,
    ) -> dict:
        """
        Generate from prompts and compute quality metrics.

        Returns dict with: repetition_ratio, distinct_2, distinct_3,
        self_bleu, consistency, all_tokens, num_tokens_generated.
        """
        prompts = self._make_prompts(num_prompts, prompt_len)

        # Generate from each prompt
        all_generated = []
        total_tokens = 0

        # Use a fixed max_new_tokens that fits in block_size
        effective_max_tokens = min(
            max_new_tokens,
            self.block_size - prompt_len - 1,
        )
        if effective_max_tokens <= 0:
            effective_max_tokens = 1

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                torch.manual_seed(42)
                result = generate_fn(
                    self.model,
                    prompt.clone().to(self.device),
                    effective_max_tokens,
                )
                if isinstance(result, torch.Tensor):
                    generated = result[0, prompt.shape[1]:].tolist()
                else:
                    generated = result[prompt.shape[1]:]

                all_generated.append(generated)
                total_tokens += len(generated)

        # Compute aggregate metrics across all generations
        all_tokens_flat = [t for gen in all_generated for t in gen]

        repetition = compute_repetition_ratio(all_tokens_flat)
        distinct_2 = compute_distinct_n(all_tokens_flat, 2)
        distinct_3 = compute_distinct_n(all_tokens_flat, 3)
        self_bleu = compute_self_bleu(all_generated)

        # Consistency: test with first prompt
        consistency = compute_consistency(
            generate_fn,
            self.model,
            prompts[0],
            effective_max_tokens,
            device=self.device,
            num_trials=num_seeds,
        )

        return {
            "repetition_ratio": repetition,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3,
            "self_bleu": self_bleu,
            "consistency": consistency,
            "all_tokens": all_generated,
            "num_tokens_generated": total_tokens,
        }

    def run_full_eval(
        self,
        generate_fn: Callable,
        implementation_name: str,
        *,
        num_prompts: int = 20,
        prompt_len: int = 16,
        max_new_tokens: int = 50,
        num_seeds: int = 3,
        num_ppl_windows: int = 50,
    ) -> EvalResult:
        """
        Full eval pipeline: perplexity + generation quality.

        Args:
            generate_fn: function(model, idx, max_new_tokens) -> tensor (1, T+N)
            implementation_name: identifier for this implementation
        """
        start_time = time.perf_counter()

        # 1. Perplexity
        ppl = self.eval_perplexity(
            num_windows=num_ppl_windows,
            window_size=min(32, self.block_size),
        )

        # 2. Generation quality
        quality = self.eval_generation_quality(
            generate_fn,
            num_prompts=num_prompts,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            num_seeds=num_seeds,
        )

        elapsed = time.perf_counter() - start_time

        return EvalResult(
            implementation=implementation_name,
            perplexity=ppl,
            repetition_ratio=quality["repetition_ratio"],
            distinct_2=quality["distinct_2"],
            distinct_3=quality["distinct_3"],
            self_bleu=quality["self_bleu"],
            consistency=quality["consistency"],
            num_prompts=num_prompts,
            num_tokens_generated=quality["num_tokens_generated"],
            eval_time_seconds=elapsed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config={
                "block_size": self.block_size,
                "vocab_size": self.vocab_size,
                "device": self.device,
                "prompt_len": prompt_len,
                "max_new_tokens": max_new_tokens,
                "num_prompts": num_prompts,
                "num_seeds": num_seeds,
            },
        )

    @staticmethod
    def compare_to_baseline(
        result: EvalResult,
        baseline: EvalResult,
        *,
        perplexity_threshold_pct: float = 5.0,
        repetition_threshold_pct: float = 10.0,
        diversity_threshold_pct: float = 10.0,
    ) -> RegressionReport:
        """
        Compare current eval against a baseline and flag regressions.

        Thresholds are percentage changes. A regression is flagged when:
          - Perplexity increases by more than `perplexity_threshold_pct`%
          - Repetition ratio increases by more than `repetition_threshold_pct`%
          - Distinct-2 or Distinct-3 decreases by more than `diversity_threshold_pct`%
          - Consistency drops below 1.0
        """
        flags = []

        def _check(
            metric_name: str,
            current: float,
            baseline_val: float,
            threshold_pct: float,
            direction: str,
        ):
            if baseline_val == 0:
                delta_pct = 0.0 if current == 0 else 100.0
            else:
                delta_pct = ((current - baseline_val) / abs(baseline_val)) * 100

            if direction == "higher_is_worse":
                is_regression = delta_pct > threshold_pct
            else:  # lower_is_worse
                is_regression = delta_pct < -threshold_pct

            flags.append(RegressionFlag(
                metric=metric_name,
                baseline_value=baseline_val,
                current_value=current,
                threshold_pct=threshold_pct,
                delta_pct=delta_pct,
                is_regression=is_regression,
                direction=direction,
            ))

        _check("perplexity", result.perplexity, baseline.perplexity,
               perplexity_threshold_pct, "higher_is_worse")
        _check("repetition_ratio", result.repetition_ratio, baseline.repetition_ratio,
               repetition_threshold_pct, "higher_is_worse")
        _check("distinct_2", result.distinct_2, baseline.distinct_2,
               diversity_threshold_pct, "lower_is_worse")
        _check("distinct_3", result.distinct_3, baseline.distinct_3,
               diversity_threshold_pct, "lower_is_worse")

        # Consistency is a hard check: must be 1.0
        consistency_regression = result.consistency < 1.0
        flags.append(RegressionFlag(
            metric="consistency",
            baseline_value=baseline.consistency,
            current_value=result.consistency,
            threshold_pct=0.0,
            delta_pct=(result.consistency - baseline.consistency) * 100,
            is_regression=consistency_regression,
            direction="lower_is_worse",
        ))

        overall_pass = not any(f.is_regression for f in flags)

        return RegressionReport(
            implementation=result.implementation,
            baseline_implementation=baseline.implementation,
            flags=flags,
            overall_pass=overall_pass,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def print_eval_result(result: EvalResult):
    """Pretty-print an eval result."""
    print(f"\n  📊 Eval: {result.implementation}")
    print(f"  {'─' * 50}")
    print(f"  Perplexity:       {result.perplexity:>10.2f}")
    print(f"  Repetition ratio: {result.repetition_ratio:>10.4f}")
    print(f"  Distinct-2:       {result.distinct_2:>10.4f}")
    print(f"  Distinct-3:       {result.distinct_3:>10.4f}")
    print(f"  Self-BLEU:        {result.self_bleu:>10.4f}")
    print(f"  Consistency:      {result.consistency:>10.2f}")
    print(f"  Tokens generated: {result.num_tokens_generated:>10d}")
    print(f"  Eval time:        {result.eval_time_seconds:>10.2f}s")


def print_regression_report(report: RegressionReport):
    """Pretty-print a regression comparison."""
    status = "✅ PASS" if report.overall_pass else "❌ REGRESSION DETECTED"
    print(f"\n  🔍 Regression Check: {report.implementation} vs {report.baseline_implementation}")
    print(f"  {'─' * 50}")
    print(f"  Overall: {status}")
    print()

    for flag in report.flags:
        icon = "✅" if not flag.is_regression else "❌"
        delta_str = f"{flag.delta_pct:+.1f}%"
        direction_hint = "↑ bad" if flag.direction == "higher_is_worse" else "↓ bad"
        print(
            f"  {icon} {flag.metric:<20s}  "
            f"baseline={flag.baseline_value:.4f}  "
            f"current={flag.current_value:.4f}  "
            f"Δ={delta_str:<8s}  "
            f"threshold=±{flag.threshold_pct:.0f}%  "
            f"({direction_hint})"
        )


def print_comparison_table(results: list[EvalResult]):
    """Print a side-by-side comparison table of multiple eval results."""
    if not results:
        return

    print(f"\n  {'=' * 70}")
    print(f"  Eval Comparison Table")
    print(f"  {'=' * 70}")

    headers = ["implementation", "ppl", "rep_ratio", "dist-2", "dist-3",
               "self_bleu", "consist", "tokens", "time_s"]
    rows = []
    for r in results:
        rows.append([
            r.implementation,
            f"{r.perplexity:.2f}",
            f"{r.repetition_ratio:.4f}",
            f"{r.distinct_2:.4f}",
            f"{r.distinct_3:.4f}",
            f"{r.self_bleu:.4f}",
            f"{r.consistency:.2f}",
            str(r.num_tokens_generated),
            f"{r.eval_time_seconds:.2f}",
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(vals):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print(f"  {fmt(headers)}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")
    for row in rows:
        print(f"  {fmt(row)}")
    print()
