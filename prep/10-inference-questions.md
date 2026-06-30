10 Rigorous LLM System Design Interview Questions

1. End-to-End LLM Serving Pipeline Design: Suppose you must design a production inference service for a 70B-parameter language model that handles hundreds of requests per second while keeping tail latency low (e.g. 95th percentile under 500ms for a moderate response length). How would you architect this system from the ground up? Consider the model deployment across hardware (GPU distribution), how you’d batch or queue requests, and what optimizations you would employ to achieve high throughput and low latency. (Tests the candidate’s ability to holistically design a large-scale LLM inference system under strict latency and throughput constraints.)

## Answer Draft 1

I would consider the following system:

Using a smaller draft model to do speculative decoding, as well as using KV caching and prefix caching. I would probably also do disaggregated prefill that mixes the prefill and decode on different processes. I would probably divide the GPU into different clusters with load balancers to direct requests. 

## Answer Draft 2

I would consider the following system:

The pipeline from the ground up will look like this:

1. Request ingestion & prefill: This is where we will do the prefilling of the KV cache. We will use dynamic batching to maximize GPU utilization. 

2. Auth rate limiting: We will use a rate limiter to limit the number of requests per second. 
This will be implemented as a service in front of the inference servers.

3. Admission control: We will use a queue where requests will wait to be admitted into the system. Requests will have a priority level assigned by the API gateway and will be served in the order of their priority. If a request has not been served for too long, it will be dropped. This deadline can be enforced by the scheduler and is a knob that can be tuned.

4. Router with session affinity: We will use a router to direct requests to the appropriate GPU. For each request, we can find out which GPU to send it to; if the session does not exist anywhere, we can send it to an available GPU using the load balancer. If the session does exist, then we will send it to the same GPU cluster. 

5. Engine abstraction: The engine abstraction will wrap the complex inference logic and provide a unified API for clients to call, which will make it easier to integrate with other systems. It will also handle things like KV caching and speculative decoding.

6. Streaming response path: The response will be streamed to the client as each token is yielded by the generation loop. This is to ensure low latency for the first token. 

7. The full observability stack (distributed tracing, metrics, alerting) will be implemented for every layer of the system.

Disaggregated prefill can be leveraged to hide the latency of the prefill stage for the decode stage. We can have separate prefill and decode engines. We can classify / hand off the request to the appropriate engine depending on whether it needs prefilling or not (can be tracked with a field in the request). 

To transfer the KV cache, we can use something like NVLink to transfer the KV cache from the prefill engine to the decode engine. We don't use shared memory here because we will be restricting ourselves to one node. Disaggregated prefill is more beneficial when we have access to multiple nodes of hardware, and when the prompts we receive are variable in length, and have workloads that are not balanced in terms of prefill vs decode. 

We will use a method like Paged Attention in vLLM and Cuda Graphs to lower the inference latency. Cuda graphs allows us to reply tensor operations that otherwise would have expensive repeated launch overhead. Tensor parallelism is important for 


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

## Answer Draft 2:

I would do this for dynamic batching mechanism:

Modern PagedAttention mechanisms are better because they avoid the overhead of padding for varying input lengths and are more memory efficient. 

Varying generation lengths: We can use a token budget per step. Split the budget between prefill and decode: e.g. 80% for prefill, 20% for decode. For decode-heavy workloads, increase the decode budget.

Continuous batching: Continuous batching allows us to preempt requests based on a priority mechanism. High priority requests will be taken off the waiting queue to be decoded or prefilled.

Scheduling will look like this:

- while there are active requests OR the waiting queue is non-empty:
    1. Check the waiting queue - can any new requests join the batch?
    2. Build the input tensor from ALL active requests (each contributes 1 token)
    3. Forward pass → get logits for all active requests at once
    4. Sample next token for each request
    5. Check: did any request hit its max_new_tokens? → remove it, emit its result
    6. Go to 1

In reality, this is very difficult to implement in prod, so Paged Attention is a better mechanism.

In order to have fairness across tenants, we can use a priority mechanism to preempt requests based on a priority mechanism. High priority requests will be taken off the waiting queue to be decoded or prefilled. If we get extremely long prompts, we can use chunked prefill to process them in batches. 

If mid-batch OOM, we will need to evict some requests from the batch. We can use a least recently used (LRU) eviction policy to evict requests from the batch. Requests that are not critical will be in the pool to be evicted. 

We will need to also have some kind of dashboard for ITL and TTFT to track and tune the budget split in production. This will involve A/B testing the different parameters to find the optimal values.



3. Key-Value Cache Utilization: In autoregressive generation, the model can cache the key/value pairs from prior tokens to avoid recomputing them on each step. How could you leverage KV caching to speed up inference in a multi-turn conversation or for repeated prompts across requests? 

Describe how you might implement a cache for previously computed states and discuss the memory vs. compute trade-offs involved. 

How would you decide when to reuse or discard cached states, and what are the challenges in managing cache consistency in a high-throughput setting? (Tests knowledge of transformer KV caching mechanics and the ability to weigh memory overhead against compute savings in practice.)

## Answer

KV caching can be used to by avoiding redundant calculations for previous tokens. We can implement a global cache for K/V pairs that can be shared across requests. However, this will increase the memory usage of the system. By implementing prefix caching, we can cache the KV pairs for the shared prefixes of requests. 

I would decide to discard cached states when the length of the cached states exceeds the context window of the model. This is because the model can only process a fixed amount of tokens at once, and if the cached states exceed this limit, the model will not be able to use them. The challenges are that we need to keep track of the length of the cached states and the number of active requests. We also need to make sure that the cached states are consistent with the current states of the model. 

4. Model Quantization and Precision Trade-offs: If GPU memory and throughput are at a premium, one option is to compress the model. Explain how you would use quantization (e.g. 8-bit or 4-bit weights) to reduce the model’s memory footprint and possibly increase inference speed. What are the impacts of lower precision on model accuracy and on hardware performance (throughput/latency)? Additionally, discuss any implementation considerations—such as quantization-aware training vs. post-training quantization or runtime decomposition techniques—and how those might affect a production inference pipeline. (Tests understanding of model compression techniques and their real-world effects on performance and accuracy in an inference setting.)

## Answer:

I would quantize specific layers like attention layers using INT8, and leave the rest of the layers in FP16. The linear layers have the most weights so they would be quantized to INT8 precision. The LayerNorm is not touched because it is sensitive to precision loss.

Lower precision can increase throughput and lower latency but at the cost of model accuracy. 

Dynamic quantization computes the activation scale **on-the-fly per input tensor**. This makes it robust to out-of-distribution inputs — you don't need calibration data, and the scale adapts to whatever activations your input produces. The tradeoff: that per-tensor scale computation adds runtime overhead.

The benefit of static post training quantization is that it is simple to implement and can be applied to existing models without retraining. It also 

5. Parallelism and Model Sharding: When a single GPU isn’t sufficient to host or compute the model, how would you split a large LLM across multiple GPUs or machines? Compare tensor/model parallelism (splitting individual layers across GPUs) with pipeline parallelism (dividing the stack of layers among GPUs in sequence) for inference. How does each approach affect latency and throughput? Discuss the challenges you’d need to address (such as synchronizing between devices, communication overhead, and load balancing) to make multi-GPU inference efficient and reliable. (Tests knowledge of distributing a model over multiple devices and the trade-offs between different parallelization strategies in terms of performance and complexity.)

## Answer:

I can split it in multiple ways. I can combine tensor parallelism with pipeline parallelism. Tensor parallelism is where we split the model across multiple GPUs by splitting the model weights. For example, if we have a model with 10 layers and 4 GPUs, we can split the model such that each GPU has 2-3 layers. Tensor parallelism is more effective for large models because it reduces the communication overhead between GPUs. In terms of throughput, tensor parallelism can be more effective for large batch sizes. In terms of latency, tensor parallelism can be more effective for small batch sizes.

Pipeline parallelism is where we split the model across multiple GPUs by splitting the model layers across GPUs in sequence. For example, if we have a model with 10 layers and 4 GPUs, we can split the model such that each GPU has 2-3 layers. Pipeline parallelism is more effective for small models because it reduces the communication overhead between GPUs. In terms of throughput, pipeline parallelism can be more effective for large batch sizes.

The challenges are the communication fabric needs to fast and efficient. In addition, the load balancers should be quickly able to adapt to the changing
topology of the cluster. We need heartbeats on each machine that will alert the distributed system about potential failures. 

6. Speculative Decoding for Faster Generation: Describe what speculative decoding is and how it can be used to accelerate LLM inference. In what scenario would you employ a speculative decoding approach, and how does it leverage a smaller “draft” model alongside the large model to reduce end-to-end latency? Explain the potential speed-ups and also the complexities or downsides of this technique (for example, managing two models, ensuring consistency of the final output, or wasted computation when the speculation is incorrect). (Tests understanding of an advanced inference optimization technique and the ability to reason about its benefits and implementation challenges.)

## Answer

Speculative decoding is using a draft model to generate candidate tokens that are then verified by the large model. If the large model agrees with the draft model, the candidate tokens are accepted and the large model can then generate more tokens. If the large model disagrees with the draft model, the candidate tokens are rejected and the large model can then generate more tokens. 

I would use speculative decoding when there is a need to boost throughput and latency for LLM inference, specifically inter token latency. 

The complexity is that you have to manage and train a separate model which can involve more GPU's. Also, the acceptance rate of the draft model depends on the training data and the distribution of the data. In addition, if the first token speculated is wrong, then all the subsequent tokens are rejected and we have to start over. This can lead to wasted computation. 

7. Memory Offloading and Management: Imagine your model and its intermediate data (like activation maps or the KV cache) don’t all fit in GPU memory during inference, especially with long context inputs. How would you design an offloading policy to move parts of the model or data to CPU memory (or even NVMe storage) and bring them back when needed? Discuss what factors you’d consider in an offloading strategy – for instance, which layers or data to offload, how to overlap data transfer with computation to hide latency, and how PCIe or interconnect bandwidth constraints come into play. What are the performance trade-offs of offloading, and how can smart scheduling minimize the impact on latency? (Tests the candidate’s grasp of memory–compute trade-offs and ability to manage limited GPU memory by trading off transfer overhead, as seen in large-model inference scenarios.)

8. Throughput vs. Latency Trade-offs: In a high-volume LLM service, you often need to maximize total throughput (tokens/sec or queries/sec) while still meeting latency requirements for individual users. How would you balance this trade-off in practice? Consider ideas like using adaptive batch sizes (batching more aggressively during peak load vs. prioritizing low latency for realtime requests), deploying separate model replicas or service tiers for high-priority low-latency requests vs. lower-priority bulk requests, or any scheduling/allocation mechanism to ensure both objectives are met. Discuss how you would evaluate the latency-throughput sweet spot and adjust the system as load patterns change. (Tests understanding of operational trade-offs in system design and the ability to devise strategies that cater to different service level objectives for throughput and latency.)

## Answer

We can definitely use aggresive batch sizes during peak load, and use techniques like request coalescing to merge multiple requests into a single batch. We can also use speculative decoding to reduce the latency of each request. We could also have model replicas in different clusters. We can have warm and cold clusters that we can scale up and down based on demand. 

I would evaluate the latency-throughput sweet spot by 

9. Fault Tolerance in Inference Pipelines: Serving large models is not only about speed – it’s also about reliability. Suppose a generation request is part-way through when a GPU server fails or a network hiccup occurs. How could you design the system to be fault-tolerant in such cases? Discuss mechanisms like checkpointing or saving intermediate state so another node could resume if possible, retrying requests from scratch (and what that means for user experience), or running duplicate inference in parallel on redundant hardware to hedge against failures. What are the pros and cons (especially in cost and complexity) of these approaches in a production, high-throughput inference environment? (Tests the candidate’s ability to incorporate reliability and failure-handling into system design, recognizing the challenges of long-running sequential processes like LLM inference.)

## Answer



10. Cost-Efficiency and Scalability Considerations: Large-scale LLM inference can be extremely expensive. What strategies would you use to optimize cost while maintaining acceptable performance? Discuss options such as using smaller or distilled models for certain tasks or routing simpler queries to cheaper models, leveraging spot instances or scale-to-zero for unused capacity, sharing GPUs across multiple models or clients (multi-tenancy) to increase utilization, and using techniques like batch processing or quantization to reduce resource usage. How would you ensure the system scales cost-effectively with demand, and what trade-offs might you have to accept to stay within budget? (Tests the candidate’s ability to think beyond pure performance and design a solution that is economically sustainable, demonstrating awareness of real-world constraints like resource cost and utilization.)

## Answer

I would have dynamic routing to route simpler queries to smaller/distilled models. In addition, I would also take use of spot instances that will be turned on during off peak hours to further optimize cost. We could also have multiple models on a single GPU and have multi-tenancy where multiple clients share the same GPU to run their models. 

Batch processing and quantization would be used to reduce resource usage. The system can scale cost effectively with demand by having a scale to zero infrastructure where the system will scale up the number of models based on the demand.  

The tradeoffs are that we may need to either compromise on model quality when necessary based on the specific nature of the request, or over provision GPU's to possibly account for higher then expected demand.
