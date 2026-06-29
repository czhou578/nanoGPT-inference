10 Rigorous LLM System Design Interview Questions

1. End-to-End LLM Serving Pipeline Design: Suppose you must design a production inference service for a 70B-parameter language model that handles hundreds of requests per second while keeping tail latency low (e.g. 95th percentile under 500ms for a moderate response length). How would you architect this system from the ground up? Consider the model deployment across hardware (GPU distribution), how you’d batch or queue requests, and what optimizations you would employ to achieve high throughput and low latency. (Tests the candidate’s ability to holistically design a large-scale LLM inference system under strict latency and throughput constraints.)

I would consider the following system:

Using a smaller draft model to do speculative decoding, as well as using KV caching. I would probably also do disaggregated prefill that mixes the prefill
and decode on different processes. I would probably divide the GPU into different clusters with load balancers 