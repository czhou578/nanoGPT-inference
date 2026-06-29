10 Rigorous LLM System Design Interview Questions

1. End-to-End LLM Serving Pipeline Design: Suppose you must design a production inference service for a 70B-parameter language model that handles hundreds of requests per second while keeping tail latency low (e.g. 95th percentile under 500ms for a moderate response length). How would you architect this system from the ground up? Consider the model deployment across hardware (GPU distribution), how you’d batch or queue requests, and what optimizations you would employ to achieve high throughput and low latency. (Tests the candidate’s ability to holistically design a large-scale LLM inference system under strict latency and throughput constraints.)

I would consider the following system:

Using a smaller draft model to do speculative decoding, as well as using KV caching and prefix caching. I would probably also do disaggregated prefill that mixes the prefill and decode on different processes. I would probably divide the GPU into different clusters with load balancers to direct requests. 

2. Dynamic Batching Strategies: LLM inference is iterative and can benefit greatly from batching multiple requests together. How would you implement a dynamic batching mechanism for incoming queries to maximize GPU utilization without introducing unacceptable latency for individual requests? Discuss how you’d handle varying input lengths and generation lengths (to avoid long prompts slowing down shorter ones), and compare approaches like fixed-size batches vs. continuous batching with fine-grained scheduling. (Tests understanding of request batching policies and how to balance throughput vs. per-request latency, e.g. via smart scheduling and grouping of requests.)

I would do this for dynamic batching mechanism:

Varying input lengths: For shorter prompts (relative to max), we pad them to the max length of the current batch. This maximizes GPU utilization. Fixed-size batches are simpler but waste capacity.

Varying generation lengths: We can use a token budget per step. Split the budget between prefill and decode: e.g. 80% for prefill, 20% for decode. For decode-heavy workloads, increase the decode budget.

Continuous batching: Continuous batching allows us to preempt requests based on a priority mechanism. High priority
requests will be taken off the waiting queue to be decoded or prefilled.

Scheduling will look like this:

- while there are active requests OR the waiting queue is non-empty:
    1. Check the waiting queue - can any new requests join the batch?
    2. Build the input tensor from ALL active requests (each contributes 1 token)
    3. Forward pass → get logits for all active requests at once
    4. Sample next token for each request
    5. Check: did any request hit its max_new_tokens? → remove it, emit its result
    6. Go to 1

3. Key-Value Cache Utilization: In autoregressive generation, the model can cache the key/value pairs from prior tokens to avoid recomputing them on each step. How could you leverage KV caching to speed up inference in a multi-turn conversation or for repeated prompts across requests? Describe how you might implement a cache for previously computed states and discuss the memory vs. compute trade-offs involved. How would you decide when to reuse or discard cached states, and what are the challenges in managing cache consistency in a high-throughput setting? (Tests knowledge of transformer KV caching mechanics and the ability to weigh memory overhead against compute savings in practice.)

KV caching can be used to by avoiding redundant calculations for previous tokens. We can implement a global cache for K/V pairs that can be shared across requests. However, this will increase the memory usage of the system. By implementing prefix caching, we can cache the KV pairs for the shared prefixes of requests. 


