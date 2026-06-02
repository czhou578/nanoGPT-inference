"""
KV-cache vs trigram speculative decoding benchmark.

This file is standalone so importing it does not train
nanogpt-trigram-spec-decode.py. It assumes the model API used by that script:

    logits, loss, new_kvs = model(idx, targets=None, pos=None,
                                  past_kvs=None, attn_mask=None,
                                  input_mask=None)

The benchmark compares:
- kv_decode: normal autoregressive decoding with a target-model KV cache
- trigram_spec_decode: a cheap trigram draft model proposes multiple tokens,
  then the target model verifies those tokens in one forward pass

The trigram draft estimates P(next_token | previous_previous, previous) from a
token corpus. It is still tiny compared with a neural draft model, but it uses
one more token of context than the bigram draft benchmark.
"""

import time
import torch

from benchmarks.speculative_decoding import (
    SpecDecodeRequestState,
    SpecDecodeRunMetrics,
    _accept_reject,
    _make_generator,
    _record_token,
    _sample_next_token,
    _sync_if_cuda,
    _target_probs_from_logits,
    _trim_kv_cache,
    make_spec_decode_workload,
    print_spec_decode_comparison_table,
    run_kv_decode_policy,
)


class TrigramDraftModel:
    """
    Cheap draft model for speculative decoding.

    It estimates P(next_token | token_{t-2}, token_{t-1}) from a token corpus.
    The optional draft_noise parameter blends the learned trigram table with a
    uniform distribution to create lower-quality draft runs.
    """

    def __init__(self, token_ids, vocab_size, device, draft_noise=0.0):
        counts = torch.ones(vocab_size, vocab_size, vocab_size, dtype=torch.float32)

        if token_ids is not None:
            ids = torch.as_tensor(token_ids, dtype=torch.long).flatten().cpu()
            if ids.numel() > 2:
                first = ids[:-2].tolist()
                second = ids[1:-1].tolist()
                third = ids[2:].tolist()
                for prev_prev, prev, next_tok in zip(first, second, third):
                    if (
                        0 <= prev_prev < vocab_size
                        and 0 <= prev < vocab_size
                        and 0 <= next_tok < vocab_size
                    ):
                        counts[prev_prev, prev, next_tok] += 1.0

        probs = counts / counts.sum(dim=-1, keepdim=True)

        if draft_noise > 0:
            noise = max(0.0, min(float(draft_noise), 1.0))
            uniform = torch.full_like(probs, 1.0 / vocab_size)
            probs = (1.0 - noise) * probs + noise * uniform

        self.probs = probs.to(device)

    def get_probs(self, prev_prev_token, prev_token, temperature=1.0):
        probs = self.probs[int(prev_prev_token), int(prev_token)]
        if temperature == 1.0:
            return probs

        scaled = probs.clamp_min(1e-12).pow(1.0 / temperature)
        return scaled / scaled.sum()

    def sample(self, prev_prev_token, prev_token, *, temperature=1.0, generator=None):
        probs = self.get_probs(
            prev_prev_token,
            prev_token,
            temperature=temperature,
        )
        token = torch.multinomial(probs, num_samples=1, generator=generator).item()
        return token, probs


def _draft_trigram_tokens(draft_model, history_tokens, k, *, temperature, generator):
    if not history_tokens:
        raise ValueError("trigram speculative decoding needs at least one history token")

    candidates = []
    draft_probs = []

    if len(history_tokens) == 1:
        prev_prev = history_tokens[-1]
        prev = history_tokens[-1]
    else:
        prev_prev = history_tokens[-2]
        prev = history_tokens[-1]

    for _ in range(k):
        next_token, probs = draft_model.sample(
            prev_prev,
            prev,
            temperature=temperature,
            generator=generator,
        )
        candidates.append(next_token)
        draft_probs.append(probs)
        prev_prev, prev = prev, next_token

    return candidates, draft_probs


@torch.no_grad()
def run_trigram_spec_decode_policy(
    model,
    workload,
    *,
    name,
    draft_model,
    speculation_len,
    device,
    temperature=1.0,
    seed=1337,
):
    if speculation_len < 1:
        raise ValueError("speculation_len must be at least 1")

    model.eval()
    generator = _make_generator(device, seed)
    states = [SpecDecodeRequestState(spec=req) for req in workload]

    _sync_if_cuda(device)
    start_time = time.perf_counter()
    forward_seconds = 0.0
    target_forward_calls = 0
    target_tokens_evaluated = 0
    verify_iterations = 0
    draft_tokens_proposed = 0
    accepted_draft_tokens = 0
    bonus_tokens = 0
    resampled_tokens = 0

    for req in states:
        req.arrived_at_s = time.perf_counter()

        prompt = torch.tensor([req.spec.prompt_tokens], dtype=torch.long, device=device)
        pos = torch.arange(len(req.spec.prompt_tokens), dtype=torch.long, device=device).unsqueeze(0)

        fwd_start = time.perf_counter()
        logits, _, req.past_kvs = model(prompt, pos=pos)
        _sync_if_cuda(device)
        forward_seconds += time.perf_counter() - fwd_start
        target_forward_calls += 1
        target_tokens_evaluated += len(req.spec.prompt_tokens)

        next_token = _sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            generator=generator,
        ).item()
        _record_token(req, next_token)

        while not req.is_done:
            remaining = req.spec.max_new_tokens - len(req.generated_tokens)
            k = min(speculation_len, remaining)
            history = req.spec.prompt_tokens + req.generated_tokens

            candidates, draft_probs = _draft_trigram_tokens(
                draft_model,
                history,
                k,
                temperature=temperature,
                generator=generator,
            )
            draft_tokens_proposed += len(candidates)

            verify_tokens = [req.last_token] + candidates
            input_ids = torch.tensor([verify_tokens], dtype=torch.long, device=device)
            cache_len = req.cache_len
            positions = torch.arange(
                cache_len,
                cache_len + len(verify_tokens),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            fwd_start = time.perf_counter()
            logits, _, new_kvs = model(input_ids, pos=positions, past_kvs=req.past_kvs)
            _sync_if_cuda(device)
            forward_seconds += time.perf_counter() - fwd_start
            target_forward_calls += 1
            target_tokens_evaluated += len(verify_tokens)
            verify_iterations += 1
            req.verify_iterations += 1

            target_probs = _target_probs_from_logits(logits, temperature=temperature)
            allow_bonus = remaining > len(candidates)
            (
                accepted,
                accepted_count,
                bonus_count,
                resampled_count,
            ) = _accept_reject(
                candidates,
                draft_probs,
                target_probs,
                allow_bonus=allow_bonus,
                generator=generator,
            )

            req.past_kvs = _trim_kv_cache(
                new_kvs,
                cache_len_before_verify=cache_len,
                keep_new_tokens=1 + accepted_count,
            )

            for token in accepted[:remaining]:
                _record_token(req, token)

            accepted_draft_tokens += accepted_count
            bonus_tokens += bonus_count
            resampled_tokens += resampled_count

        req.completed_at_s = time.perf_counter()

    _sync_if_cuda(device)
    end_time = time.perf_counter()

    return SpecDecodeRunMetrics(
        name=name,
        total_requests=len(states),
        total_prompt_tokens=sum(len(req.spec.prompt_tokens) for req in states),
        total_generated_tokens=sum(len(req.generated_tokens) for req in states),
        total_seconds=end_time - start_time,
        request_latencies_s=[req.completed_at_s - req.arrived_at_s for req in states],
        ttft_s=[req.first_token_at_s - req.arrived_at_s for req in states],
        forward_seconds=forward_seconds,
        target_forward_calls=target_forward_calls,
        target_tokens_evaluated=target_tokens_evaluated,
        verify_iterations=verify_iterations,
        draft_tokens_proposed=draft_tokens_proposed,
        accepted_draft_tokens=accepted_draft_tokens,
        bonus_tokens=bonus_tokens,
        resampled_tokens=resampled_tokens,
    )


def run_kv_vs_trigram_speculative_decoding_benchmark(
    model,
    *,
    vocab_size,
    training_token_ids=None,
    prompt_source_tokens=None,
    num_requests=8,
    prompt_len=16,
    max_new_tokens=12,
    speculation_len=4,
    draft_noise=0.0,
    device=None,
    seed=1337,
    temperature=1.0,
):
    if device is None:
        device = next(model.parameters()).device

    workload = make_spec_decode_workload(
        vocab_size=vocab_size,
        num_requests=num_requests,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        prompt_source_tokens=prompt_source_tokens,
        seed=seed,
    )

    draft_model = TrigramDraftModel(
        training_token_ids,
        vocab_size=vocab_size,
        device=device,
        draft_noise=draft_noise,
    )

    baseline = run_kv_decode_policy(
        model,
        workload,
        name="kv_decode",
        device=device,
        temperature=temperature,
        seed=seed,
    )

    trigram_spec = run_trigram_spec_decode_policy(
        model,
        workload,
        name=f"trigram_spec_decode_k{speculation_len}",
        draft_model=draft_model,
        speculation_len=speculation_len,
        device=device,
        temperature=temperature,
        seed=seed,
    )

    print_spec_decode_comparison_table([baseline, trigram_spec])

    return {
        "kv_decode": baseline,
        "trigram_spec_decode": trigram_spec,
    }
