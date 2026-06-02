# Trigram Speculative Decoding Benchmark Results Write-up

This note explains the results in [`results/trigram_spec_decode_results.txt`](../../results/trigram_spec_decode_results.txt) for the KV-cache decoding vs trigram speculative decoding benchmark implemented in [`benchmarks/trigram_speculative_decoding.py`](../../benchmarks/trigram_speculative_decoding.py) and configured by [`benchmarks/trigram_speculative_decoding_benchmark_runs.py`](../../benchmarks/trigram_speculative_decoding_benchmark_runs.py).

## Executive Summary

This benchmark compares two decoding paths:

- **`kv_decode`**: normal autoregressive decoding with the target model and a KV cache. After prefill, each target forward produces one generated token.
- **`trigram_spec_decode`**: a cheap trigram draft model proposes several future tokens, and the target model verifies those candidates in a larger forward pass.

The trigram speculative path is faster in every workload:

- Throughput improves by **1.40x to 2.13x**.
- Average request latency drops by about **29% to 53%**.
- Target forward calls fall to **47% to 62%** of the KV baseline.
- TTFT is slightly better in every row.

The main tradeoff is that speculative decoding evaluates more target tokens:

- Target token work rises to **1.29x to 1.70x** of the KV baseline.
- Acceptance is modest, ranging from **25.7% to 39.9%**.
- Larger speculation depths reduce target calls, but they also create more rejected or wasted speculative positions.

The core result:

> Trigram speculative decoding improves throughput because it greatly reduces the number of target-model forward calls, even though it evaluates more total target tokens than normal KV decoding.

This is the same fundamental speculative-decoding tradeoff seen in the bigram benchmark, but with a trigram draft model: fewer serial target calls, larger verification calls, and a draft acceptance rate that determines how much of the speculative work becomes useful output.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CPU
- **Context length:** `block_size=64`
- **Prompt length:** `24` tokens in every row
- **Draft model:** smoothed trigram table built from training-token transition counts
- **Benchmark target:** speculative decoding mechanics, not output quality

The tiny model trains briefly before the benchmark:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8682 |
| 119 | 2.7759 | 2.7995 |

The generated sample text is noisy, which is expected for a tiny character-level model trained for a short run. The benchmark is focused on serving behavior: target forward calls, draft acceptance, verification size, latency, and throughput.

## What Was Benchmarked

The benchmark runs the same request workload through two policies.

| Method | Behavior |
|---|---|
| `kv_decode` | Prefill the prompt once, sample the first token, then decode one token per target-model forward using the KV cache. |
| `trigram_spec_decode` | Prefill the prompt once, sample the first token, then use a trigram draft model to propose `K` future tokens. The target model verifies `[current_token] + candidates` in one forward pass. |

The trigram draft estimates:

```text
P(next_token | token_{t-2}, token_{t-1})
```

That means it conditions on the previous two tokens instead of only the previous one. It is still extremely cheap compared with a neural draft model, but it has more context than a bigram table.

The speculative loop has three phases:

1. **Draft:** propose up to `K` candidate tokens from the trigram table.
2. **Verify:** run the target model on the current token plus all candidates.
3. **Accept/reject:** accept matching draft tokens, sample a corrected token on rejection, or sample a bonus token if the full draft chain is accepted.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests served. |
| `prompt_tok` | Total prompt tokens processed during prefill. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `target_calls` | Number of target-model forward calls. Lower is usually better. |
| `target_tok` | Total tokens evaluated by the target model, including prompt and verification tokens. |
| `tgt_tok/gen` | Target tokens evaluated per generated token. |
| `avg_verify` | Average emitted tokens per speculative verification step. Higher is better. |
| `accept` | Fraction of proposed draft tokens accepted by target verification. |
| `draft_tok` | Number of draft tokens proposed. |
| `bonus` | Number of bonus tokens sampled when every draft candidate in a step is accepted. |
| `resample` | Number of corrected tokens sampled after rejection. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `forward_s` | Time spent in measured target-model forward work. |

For this benchmark, the most important metrics are `gen_tok/s`, `target_calls`, `target_tok`, `avg_verify`, `accept`, and `avg_lat_ms`.

## Results Summary

| Case | Requests | Generated Tokens | K | Draft Noise | KV Tok/s | Trigram Spec Tok/s | Throughput Ratio | Target Call Ratio | Target Token Ratio | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k2_trigram_draft` | 8 | 128 | 2 | 0.0 | 545.44 | 1162.72 | 2.13x | 0.62x | 1.29x | 39.9% |
| `k4_trigram_draft` | 8 | 128 | 4 | 0.0 | 877.26 | 1232.47 | 1.40x | 0.54x | 1.52x | 28.1% |
| `k6_trigram_draft` | 8 | 128 | 6 | 0.0 | 959.98 | 1361.51 | 1.42x | 0.48x | 1.70x | 25.7% |
| `k4_noisy_trigram` | 8 | 128 | 4 | 0.5 | 941.86 | 1318.78 | 1.40x | 0.51x | 1.47x | 31.9% |
| `longer_outputs_k4_trigram` | 6 | 144 | 4 | 0.0 | 958.46 | 1431.82 | 1.49x | 0.47x | 1.58x | 33.5% |

Speculative decoding improves throughput in every row. The average throughput ratio is about **1.57x**.

One caution: the `k2_trigram_draft` baseline is much slower than the other KV baselines in this file. That makes its **2.13x** ratio look especially strong. The absolute speculative throughput for K=2 is not the fastest; K=6 and the longer-output K=4 run are faster in absolute generated tokens per second.

## Main Trend: The Trigram Draft Reduces Target Calls

The cleanest success signal is target forward calls.

| Case | KV Target Calls | Trigram Target Calls | Calls Saved | Reduction |
|---|---:|---:|---:|---:|
| `k2_trigram_draft` | 128 | 79 | 49 | 38.3% |
| `k4_trigram_draft` | 128 | 69 | 59 | 46.1% |
| `k6_trigram_draft` | 128 | 61 | 67 | 52.3% |
| `k4_noisy_trigram` | 128 | 65 | 63 | 49.2% |
| `longer_outputs_k4_trigram` | 144 | 68 | 76 | 52.8% |

This is exactly what speculative decoding is meant to do. Normal KV decode has a serial loop:

```text
target forward -> one generated token
target forward -> one generated token
target forward -> one generated token
...
```

Trigram speculative decoding changes that shape:

```text
draft proposes several tokens
target verifies several positions in one call
emit one or more tokens
```

Even when many draft tokens are rejected, each verification step can still emit a corrected token. When some tokens are accepted, the request advances by more than one token from one target call.

## The Tradeoff: More Target Tokens Evaluated

The speedup does not come from reducing the total number of target tokens evaluated. It comes from reducing the number of separate target calls.

| Case | KV Target Tokens | Trigram Target Tokens | Increase |
|---|---:|---:|---:|
| `k2_trigram_draft` | 312 | 401 | +28.5% |
| `k4_trigram_draft` | 312 | 474 | +51.9% |
| `k6_trigram_draft` | 312 | 529 | +69.6% |
| `k4_noisy_trigram` | 312 | 459 | +47.1% |
| `longer_outputs_k4_trigram` | 282 | 445 | +57.8% |

This is the central speculative-decoding exchange:

- **Cost:** target verification evaluates extra candidate positions.
- **Benefit:** those positions are evaluated in fewer, larger forward calls.

On this CPU run, fewer calls win. The target token count rises substantially, but wall-clock time still drops.

## Speculation Depth: K=2 vs K=4 vs K=6

The three clean trigram rows compare speculation depth on the same 8-request, 128-token workload.

| Case | K | Spec Tok/s | Target Calls | Target Tokens | Avg Verify | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| `k2_trigram_draft` | 2 | 1162.72 | 79 | 401 | 1.69 | 39.9% |
| `k4_trigram_draft` | 4 | 1232.47 | 69 | 474 | 1.97 | 28.1% |
| `k6_trigram_draft` | 6 | 1361.51 | 61 | 529 | 2.26 | 25.7% |

Increasing `K` reduces target calls:

- K=2 uses **79** target calls.
- K=4 uses **69** target calls.
- K=6 uses **61** target calls.

It also increases average emitted tokens per verification:

- K=2 emits **1.69** tokens per verify step.
- K=4 emits **1.97** tokens per verify step.
- K=6 emits **2.26** tokens per verify step.

But this comes with two costs:

- Acceptance drops from **39.9%** to **25.7%**.
- Target tokens evaluated rise from **401** to **529**.

The best absolute throughput among these three rows is K=6 at **1361.51 tok/s**, but K=6 is also doing the most extra target-token work. This is the classic speculation-depth tradeoff:

> Larger K can reduce serial target calls, but weak later predictions create more rejected candidate work.

## Acceptance Rate Is Modest

The trigram draft's acceptance rates are not high:

| Case | Draft Tokens Proposed | Accepted Draft Tokens | Acceptance |
|---|---:|---:|---:|
| `k2_trigram_draft` | 138 | 55 | 39.9% |
| `k4_trigram_draft` | 221 | 62 | 28.1% |
| `k6_trigram_draft` | 284 | 73 | 25.7% |
| `k4_noisy_trigram` | 210 | 67 | 31.9% |
| `longer_outputs_k4_trigram` | 239 | 80 | 33.5% |

This means most proposed draft tokens are not directly accepted. The speculative path still wins because it often emits more than one token per target verification and reduces the number of target calls.

Still, the acceptance rate sets the ceiling. A stronger draft model would likely:

- increase `avg_verify`,
- reduce `resample`,
- reduce target tokens evaluated per generated token,
- make larger `K` values more efficient.

The low acceptance also shows that the trigram draft is not a very strong approximation of the target model. It is useful as a cheap educational draft, but it is not close to a production neural draft model.

## Bonus And Resampled Tokens

Two counters explain how speculative steps end.

| Case | Bonus Tokens | Resampled Tokens |
|---|---:|---:|
| `k2_trigram_draft` | 15 | 50 |
| `k4_trigram_draft` | 4 | 54 |
| `k6_trigram_draft` | 2 | 45 |
| `k4_noisy_trigram` | 7 | 46 |
| `longer_outputs_k4_trigram` | 6 | 52 |

**Bonus tokens** happen when all draft candidates in a verification step are accepted. The target has already computed the next distribution, so the benchmark can sample one extra token.

**Resampled tokens** happen when a draft candidate is rejected. The benchmark emits a corrected token and stops using the rest of that speculative batch.

The pattern is intuitive:

- K=2 has the most bonus tokens because shorter speculative chains are easier to accept completely.
- K=4 and K=6 have fewer bonus tokens because a longer chain is more likely to reject somewhere.
- Resampling is common in every trigram run, which reflects the modest draft quality.

## Latency And TTFT

Trigram speculative decoding improves average latency in every row:

| Case | KV Avg Latency | Trigram Avg Latency | Reduction |
|---|---:|---:|---:|
| `k2_trigram_draft` | 29.33 ms | 13.76 ms | 53.1% |
| `k4_trigram_draft` | 18.24 ms | 12.98 ms | 28.8% |
| `k6_trigram_draft` | 16.67 ms | 11.75 ms | 29.5% |
| `k4_noisy_trigram` | 16.99 ms | 12.13 ms | 28.6% |
| `longer_outputs_k4_trigram` | 25.04 ms | 16.76 ms | 33.1% |

Latency improves because each request needs fewer target-model iterations to reach its output length.

TTFT also improves slightly:

| Case | KV Avg TTFT | Trigram Avg TTFT |
|---|---:|---:|
| `k2_trigram_draft` | 2.00 ms | 1.30 ms |
| `k4_trigram_draft` | 1.72 ms | 1.37 ms |
| `k6_trigram_draft` | 1.51 ms | 1.29 ms |
| `k4_noisy_trigram` | 1.65 ms | 1.33 ms |
| `longer_outputs_k4_trigram` | 1.63 ms | 1.36 ms |

TTFT is not the main speculative-decoding benefit because both methods still prefill the prompt before emitting the first sampled token. The larger win is in post-first-token decode latency.

## The Noisy Trigram Result

`k4_noisy_trigram` uses the same K=4 depth as `k4_trigram_draft`, but blends the draft distribution with uniform noise:

```text
draft_noise=0.5
```

Results:

- Throughput rises from **1232.47** to **1318.78 tok/s** compared with clean K=4.
- Target calls fall from **69** to **65**.
- Target tokens fall from **474** to **459**.
- Acceptance rises from **28.1%** to **31.9%**.

This looks like noise helped, but it should be interpreted carefully. These are short, stochastic runs. Sampling variation can easily move acceptance and latency. The safer reading is:

> The K=4 trigram speculative path is robust to a noisier draft distribution in this run.

To prove that noise truly improves the draft, this benchmark would need multiple seeds, larger sample sizes, and direct comparison against held-out next-token accuracy.

## Longer Output Run

The `longer_outputs_k4_trigram` row changes the workload to:

```text
num_requests=6
prompt_len=24
max_new_tokens=24
speculation_len=4
```

This creates more decode work per request. That matters because speculative decoding is mainly a decode-phase optimization.

Results:

- KV throughput: **958.46 tok/s**
- Trigram speculative throughput: **1431.82 tok/s**
- Throughput ratio: **1.49x**
- Target calls drop from **144** to **68**
- Average latency drops from **25.04 ms** to **16.76 ms**

This is one of the most meaningful rows. Longer outputs give speculative decoding more opportunities to collapse many one-token decode forwards into fewer verification forwards.

## Forward Time

The measured forward time drops in every row:

| Case | KV Forward Time | Trigram Forward Time | Reduction |
|---|---:|---:|---:|
| `k2_trigram_draft` | 0.2227 s | 0.0947 s | 57.5% |
| `k4_trigram_draft` | 0.1387 s | 0.0871 s | 37.2% |
| `k6_trigram_draft` | 0.1269 s | 0.0774 s | 39.0% |
| `k4_noisy_trigram` | 0.1295 s | 0.0819 s | 36.8% |
| `longer_outputs_k4_trigram` | 0.1430 s | 0.0839 s | 41.3% |

This confirms that the speedup is not just outside the model. The speculative path reduces synchronized target-forward time despite evaluating more target tokens, because it uses far fewer separate target calls.

## Row-by-Row Interpretation

### `k2_trigram_draft`

This is the conservative trigram setting:

```text
speculation_len=2
draft_noise=0.0
```

Results:

- Throughput improves from **545.44** to **1162.72 tok/s**.
- Target calls drop from **128** to **79**.
- Acceptance is **39.9%**, the highest in the suite.
- Average latency drops from **29.33 ms** to **13.76 ms**.

This row has the largest throughput ratio, but the KV baseline is unusually slow compared with the other KV rows. The result still clearly shows speculative decoding working, but the ratio should not be over-weighted.

### `k4_trigram_draft`

This row increases speculation depth to 4:

```text
speculation_len=4
draft_noise=0.0
```

Results:

- Throughput improves from **877.26** to **1232.47 tok/s**.
- Target calls drop from **128** to **69**.
- Target tokens rise from **312** to **474**.
- Acceptance falls to **28.1%**.

K=4 is a more aggressive setting. It saves more target calls than K=2, but it also verifies many more candidate positions and accepts a smaller fraction of them.

### `k6_trigram_draft`

This row pushes speculation depth to 6:

```text
speculation_len=6
draft_noise=0.0
```

Results:

- Throughput reaches **1361.51 tok/s**, the fastest 128-token clean trigram run.
- Target calls drop to **61**, the lowest among the 128-token clean rows.
- Target tokens rise to **529**, the highest among the same rows.
- Acceptance falls to **25.7%**.

K=6 wins absolute throughput here, but it is not a free win. It does much more target-token verification work. On a larger model or different hardware, that extra work might become more expensive.

### `k4_noisy_trigram`

This row adds noise to the K=4 trigram draft:

```text
speculation_len=4
draft_noise=0.5
```

Results:

- Throughput is **1318.78 tok/s**.
- Target calls drop to **65**.
- Acceptance is **31.9%**.
- Average latency is **12.13 ms**.

The noisy row performs better than the clean K=4 row in this single run, but the likely explanation is sampling variance. The stronger conclusion is that the trigram speculative mechanism remains effective even when the draft distribution is imperfect.

### `longer_outputs_k4_trigram`

This row tests K=4 on longer outputs:

```text
num_requests=6
max_new_tokens=24
```

Results:

- Throughput improves from **958.46** to **1431.82 tok/s**.
- Target calls drop from **144** to **68**.
- Average emitted tokens per verification is **2.23**.
- Average latency drops by about **33%**.

This row confirms that trigram speculative decoding remains useful when decode work is a larger share of the request.

## Why Trigram Speculative Decoding Matters

KV caching avoids recomputing the full prompt at every decode step, but it does not remove the serial autoregressive loop. A normal decoder still asks the target model for one token at a time.

Speculative decoding uses a cheap draft model to guess several future tokens, then asks the target model to verify those guesses in one call. The target model still controls the final distribution, but the serving loop may advance by multiple tokens per verification.

This benchmark shows that even a simple trigram draft can provide useful speedups:

- target calls are roughly halved,
- latency drops,
- generated-token throughput improves in every run,
- longer decode workloads benefit clearly.

## Caveats

These results are useful, but they are not a production benchmark.

1. **The model is tiny.** With only `0.056769M` parameters, Python and PyTorch overhead can strongly affect timing.
2. **The run is CPU-only.** GPU behavior may differ, especially for larger target models and larger verification batches.
3. **The draft is a smoothed trigram table.** It is cheap, but not a strong approximation of the target model.
4. **Acceptance is low.** Most proposed draft tokens are rejected, so a better draft could improve the result substantially.
5. **The benchmark is stochastic.** Single-run differences, especially the noisy-draft row and the slow K=2 baseline, should be interpreted carefully.
6. **Quality is not measured.** The benchmark measures speed and mechanics, not generated text quality.

## Practical Takeaways

1. **The trigram speculative path works.** It completes the same generated-token count and improves throughput in every case.
2. **The main win is fewer target calls.** Target calls fall to **47% to 62%** of normal KV decoding.
3. **Target-token work increases.** The speculative path evaluates **29% to 70%** more target tokens.
4. **Acceptance is the limiting factor.** Acceptance ranges only **25.7% to 39.9%**, which leaves plenty of room for a stronger draft.
5. **Larger K is a tradeoff.** K=6 has the fastest clean trigram throughput, but also the most extra target-token verification.
6. **Longer outputs are a natural fit.** The longer-output row shows a clear speedup when decode work dominates more of the request.

## Suggested Follow-up Benchmarks

Useful next experiments would be:

- Run multiple seeds and report mean/stddev for throughput and acceptance.
- Compare trigram results directly against the bigram speculative benchmark on the same run.
- Add a held-out draft accuracy metric for bigram vs trigram next-token prediction.
- Try `K=8` to find where extra verification work overwhelms saved calls.
- Increase model size and run on CUDA to see whether the optimal K changes.
- Track draft sampling time separately from target verification time.
- Compare a neural draft model against the trigram table.

## Bottom Line

Trigram speculative decoding is effective in this benchmark. It improves generated-token throughput by about **1.6x on average**, cuts target-model calls roughly in half, and reduces request latency in every workload.

The result should be read carefully, though: the trigram draft has low acceptance and evaluates many extra target tokens. The speedup comes from changing the shape of target work, not from reducing total target-token work.

In short:

> The trigram draft is cheap enough and useful enough to reduce serial decode calls, but not accurate enough to avoid substantial rejected verification work.
