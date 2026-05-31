import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random

class BenchMarkConfig:
    def __init__(self):
        self.prompt_tokens = 128
        self.max_new_tokens = 256
        self.K = 4 #speculative
        self.batch_size = 4
        self.chunk_size = 16
        self.continuous_batching = True
        self.prefix_caching = True
        self.paged_kv = True
        self.priority_scheduling = True
        self.target_quantization = 'fp32'
        self.draft_quantization = 'fp32'
        self.token_budget = 16
        self.schedule = 'FCFS'
        self.block_size = 16
        self.kv_cache_size = 128

    def get_config(self):
        return self.__dict__    

class BenchmarkMetrics:
    def __init__(self):
        self.total_wall_time = 0
        self.total_generated_tokens = 0
        self.tokens_per_sec = 0
        self.request_latency = 0
        self.time_to_first_token = 0
        self.inter_token_latency = 0
        self.prefill_tokens_per_sec = 0
        self.decode_tokens_per_sec = 0
        self.batch_size_per_step = 0
        self.tokens_per_forward = 0
        self.kv_cache_tokens = 0
        self.allocated_kv_blocks = 0
        self.prefix_cache_hits = 0
        self.speculative_acceptance_rate = 0
        self.target_forwards_avoided = 0

    def __str__(self):
        return f"BenchmarkMetrics({self.__dict__})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict__

@dataclass
class RequestSpec:
    id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    arrival_step: int = 0
    priority: int = 0
    group: Optional[str] = None

class WorkloadGenerator:
    def __init__(self, vocab_size: int, seed: int = 1337):
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def random_tokens(self, length: int) -> List[int]:
        return [self.rng.randrange(self.vocab_size) for _ in range(length)]

    def make_uniform_batch(
        self,
        num_requests: int,
        prompt_len: int,
        max_new_tokens: int,
        arrival_gap: int = 0,
    ) -> List[RequestSpec]:
        requests = []

        for i in range(num_requests):
            requests.append(
                RequestSpec(
                    id=i,
                    prompt_tokens=self.random_tokens(prompt_len),
                    max_new_tokens=max_new_tokens,
                    arrival_step=i * arrival_gap,
                    priority=0,
                    group="uniform",
                )
            )

        return requests

    def make_mixed_lengths(
        self,
        num_requests: int,
        prompt_lens=(4, 8, 16, 32),
        output_lens=(8, 16, 32, 64),
        max_arrival_step: int = 20,
    ) -> List[RequestSpec]:
        requests = []

        for i in range(num_requests):
            prompt_len = self.rng.choice(prompt_lens)
            output_len = self.rng.choice(output_lens)

            requests.append(
                RequestSpec(
                    id=i,
                    prompt_tokens=self.random_tokens(prompt_len),
                    max_new_tokens=output_len,
                    arrival_step=self.rng.randint(0, max_arrival_step),
                    priority=0,
                    group="mixed",
                )
            )

        return sorted(requests, key=lambda r: r.arrival_step)        

    def make_prefix_shared(
        self,
        num_requests: int,
        shared_prefix_len: int,
        unique_suffix_len: int,
        max_new_tokens: int,
        num_groups: int = 2,
    ) -> List[RequestSpec]:
        requests = []
        shared_prefixes = {
            g: self.random_tokens(shared_prefix_len)
            for g in range(num_groups)
        }

        for i in range(num_requests):
            group_id = i % num_groups
            prompt = (
                shared_prefixes[group_id]
                + self.random_tokens(unique_suffix_len)
            )

            requests.append(
                RequestSpec(
                    id=i,
                    prompt_tokens=prompt,
                    max_new_tokens=max_new_tokens,
                    arrival_step=i,
                    priority=0,
                    group=f"prefix_group_{group_id}",
                )
            )

        return requests

    def make_priority_mix(
        self,
        num_requests: int,
        prompt_len: int,
        max_new_tokens: int,
        high_priority_fraction: float = 0.25,
    ) -> List[RequestSpec]:
        requests = []

        for i in range(num_requests):
            is_high_priority = self.rng.random() < high_priority_fraction

            requests.append(
                RequestSpec(
                    id=i,
                    prompt_tokens=self.random_tokens(prompt_len),
                    max_new_tokens=max_new_tokens,
                    arrival_step=i,
                    priority=0 if is_high_priority else 10,
                    group="priority_mix",
                )
            )

        return requests  

workloads = {
    "single_short": gen.make_uniform_batch(
        num_requests=1,
        prompt_len=8,
        max_new_tokens=32,
    ),

    "batch_mixed_lengths": gen.make_mixed_lengths(
        num_requests=32,
        prompt_lens=(4, 8, 16, 32),
        output_lens=(8, 16, 32, 64),
        max_arrival_step=16,
    ),

    "prefix_shared": gen.make_prefix_shared(
        num_requests=32,
        shared_prefix_len=16,
        unique_suffix_len=8,
        max_new_tokens=32,
        num_groups=4,
    ),

    "priority_mix": gen.make_priority_mix(
        num_requests=32,
        prompt_len=16,
        max_new_tokens=32,
        high_priority_fraction=0.25,
    ),
}      
