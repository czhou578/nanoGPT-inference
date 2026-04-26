---
title: "Dynamic Batching in LLM Inference Explained"
source: "https://grok.com/c/702cd182-6bd6-4bc8-ab1b-c13de5e1822e?rid=456913a4-99c6-4906-9c2c-4600f1c2fdb3"
author:
published:
created: 2026-04-25
description: "Grok is an AI assistant built by xAI. Chat, create images, write code, and get real-time answers from the web and X."
tags:
  - "clippings"
---
could you explain dynamic batching in llm inference from first principles?

Dynamic batching (also called **continuous batching** or **iteration-level scheduling**) is a technique used in LLM inference serving systems to maximize hardware utilization and throughput when handling many concurrent user requests. I'll explain it **from first principles**, starting with the absolute basics of how an LLM actually runs during inference, then building up to why static batching is inefficient, and finally how dynamic batching fixes it.

### 1\. LLM inference at the lowest level (autoregressive generation)

An LLM (e.g., a transformer decoder like Llama or Grok) generates text **one token at a time**:

- You give it a prompt of length $P$ tokens.
- It outputs token $t_1$ .
- It then takes the prompt + $t_1$ to output $t_2$ , and so on, until it hits an end-of-sequence token or a length limit.

Naïve forward pass every time would be extremely slow (you'd re-compute the entire prompt for every new token). So we use **KV caching**:

- During the first pass over the prompt (**prefill phase**), the model computes and **stores** the Key and Value vectors for every past token in a cache (size roughly $2 \times \text{layers} \times \text{heads} \times \text{dim} \times P$ ).
- For every subsequent token (**decode phase**), the model only computes the attention for the *new* token, reading the cached KV values from all previous positions. The compute cost per new token becomes essentially constant (independent of prompt length).

Mathematically, at decode step $t$ , the attention operation looks like:

$$
\text{Attention}(Q_{\text{new}}, K_{\text{cached}+ \text{new}}, V_{\text{cached}+ \text{new}})
$$

where only the new $Q, K, V$ are computed; everything older is reused.

This is already much faster, but we still have a problem when serving *many* users at once.

### 2\. Why batching matters at all

Modern GPUs (and TPUs) are massively parallel. A single matrix multiplication (the core of every transformer layer) runs *much* faster when the batch dimension is large because:

- More data can be processed in parallel across thousands of CUDA cores.
- Memory bandwidth is amortized better.
- The GPU stays busy instead of idling.

If you run one request at a time, your GPU might be using only 5–10 % of its capacity. Batching = stacking multiple independent sequences along the batch dimension so the GPU can crunch them *simultaneously*.

### 3\. Static (traditional) batching — the simple but wasteful approach

The obvious way:

1. Wait until you have exactly $B$ requests (e.g., batch size 32).
2. Prefill all $B$ prompts together (pad shorter ones to the longest prompt length with a mask so attention ignores padding).
3. Run decode steps *in lockstep*: every step you compute the next token for all $B$ sequences at once.
4. When *any* sequence finishes (EOS or max tokens), you have two bad choices:
	- Keep it in the batch with dummy padding → you waste compute and memory on a finished request.
		- Stop the whole batch early, return the finished requests, and start a *new* batch with the remaining ones (plus any new arrivals). This causes GPU idle time while waiting for enough new requests.

**Problems from first principles**:

- Prompt lengths vary wildly (some users send 10 tokens, others 2000).
- Generation lengths vary wildly (some stop after 5 tokens, others want 500+).
- Requests arrive at unpredictable times (real users don’t line up neatly).
- Result: average batch size is much smaller than the hardware’s sweet spot, GPU utilization drops, throughput collapses, and users wait longer in queues.

This is how many early inference servers worked (and why they felt slow).

### 4\. Dynamic batching — the fix (continuous/iteration-level scheduling)

The core insight: **the batch membership does not have to be fixed for the entire lifetime of the requests**. We let requests *join and leave* the active batch *between every single decode step*.

Here’s how it works from first principles:

- The inference engine maintains two pools:
	- **Waiting queue**: newly arrived prompts that haven’t been prefilled yet.
		- **Active decoding batch**: up to $B_{\max}$ sequences that have already been prefilled and are ready to generate the next token. Each has its own KV cache (of varying lengths).
- At every iteration (one GPU forward pass):
	1. **Decode step**: Take *all* (or up to $B_{\max}$ ) currently active sequences and run **one parallel decode step**. You get one new token per active sequence.
		2. **Cleanup**: For any sequence that just generated EOS or hit its max length → mark it finished, free its KV cache, and remove it from the active batch. This instantly frees a slot.
		3. **Prefill new requests**: If the active batch now has free slots *and* there are waiting prompts, grab as many as will fit (up to $B_{\max}$ ), run a (possibly smaller) prefill pass on them, and add the resulting sequences to the active batch.
- Crucially: the batch size is **re-evaluated every single token generation step**. If a request finishes, the next iteration automatically runs with whatever is left + any new arrivals. No waiting, no padding of finished sequences.

Pseudocode (simplified):

```markdown
active_batch = []          # list of sequences with KV caches
waiting_queue = []         # new prompts

while True:
    # 1. Decode current active batch (GPU is fully utilized)
    if active_batch:
        next_tokens = model.decode_step(active_batch)   # batch size = len(active_batch)
        for seq in active_batch:
            seq.append(next_tokens[seq.id])
            if seq.is_done():
                active_batch.remove(seq)                 # ← dynamic removal!

    # 2. Prefill any new requests that fit
    slots_free = B_max - len(active_batch)
    if slots_free > 0 and waiting_queue:
        new_requests = waiting_queue.pop_up_to(slots_free)
        prefilled = model.prefill_batch(new_requests)    # can be small batch
        active_batch.extend(prefilled)                   # ← dynamic addition!
```

### 5\. Why this is dramatically better

- **GPU utilization stays near 100 %**: The batch size stays close to $B_{\max}$ almost all the time, because finished sequences are removed *immediately* and new ones are added *immediately*.
- **Throughput skyrockets**: Real-world systems see 2–5× higher tokens/second compared to static batching.
- **Latency improves**: A new user request doesn’t have to wait for a full batch of $B$ other users to finish; it can be prefilled and inserted as soon as a slot opens.
- **Handles real-world messiness**: Variable prompt lengths, variable generation lengths, and asynchronous arrivals are no longer a problem.

Modern systems take this even further with **PagedAttention** (vLLM) or equivalent memory management, treating the KV cache like virtual memory pages that can be allocated/deallocated dynamically, but the *scheduling* principle above is the heart of dynamic batching.

In short: static batching treats the GPU like a rigid assembly line; dynamic batching treats it like a flexible, always-busy highway where cars (requests) enter and exit lanes independently at every step. That’s why almost every high-performance LLM serving framework today (vLLM, Hugging Face TGI, TensorRT-LLM, etc.) uses some form of it.

could you explain continuous batching from first principles and how it relates to dynamic batching?

**Continuous batching** (also known as **iteration-level scheduling** or **in-flight batching**) is the dominant technique used in modern LLM serving systems (vLLM, Hugging Face TGI, TensorRT-LLM, etc.) to achieve high throughput under real-world workloads with variable prompt lengths, variable output lengths, and asynchronous request arrivals.

It builds directly on the autoregressive nature of LLM generation and **KV caching**, which I explained in the previous response on dynamic batching. Here, I'll derive it **from first principles**, step by step, and then clarify how it relates to (and often overlaps with) the term "dynamic batching."

### 1\. Starting from the fundamentals: One request at a time

An LLM generates text **autoregressively** — one token at a time:

- **Prefill phase**: Process the entire prompt (say, 500 tokens) in one forward pass. This computes and stores **KV caches** for all prompt tokens. Cost scales with prompt length.
- **Decode phase**: For each new token, run a forward pass using only the *new* token's query + the cached KVs from all previous tokens. The per-token cost is roughly constant (much cheaper than re-processing the whole context).

Without batching, a GPU is terribly underutilized during decode because each forward pass processes just one sequence.

### 2\. The naive solution: Static (fixed) batching

Group N requests into a batch, process the whole batch together from start to finish, then start the next batch.

Problems (especially bad for LLMs):

- Requests have wildly different prompt lengths and output lengths.
- During decode, shorter sequences finish early but the batch stays "alive" until the longest one finishes → you waste compute/memory on padding or dummy work.
- After a batch finishes, the GPU may sit idle while waiting for enough new requests to form the next full batch.
- Result: Low average batch size, low GPU utilization, poor throughput, and higher latency (new requests wait for the current batch to complete).

### 3\. Improving grouping: What people often call "dynamic batching"

A better grouping strategy: Don't wait for a perfectly full batch of fixed size. Instead, collect incoming requests over a short time window (e.g., 10–100 ms) or until you hit a maximum batch size/memory limit, then launch whatever you have.

This is sometimes called **dynamic batching** or **request-level dynamic batching**:

- It reduces queuing delay for early requests.
- The batch still runs to completion as a unit (all sequences stay until the slowest one finishes).
- It's a clear improvement over pure static batching, especially for moderate traffic.

However, for LLMs it still leaves significant inefficiency on the table because **output lengths vary so much**. The GPU still idles (or pads) while waiting for the longest sequence in the current batch.

### 4\. The key insight for continuous batching (iteration-level scheduling)

LLM decode is **iterative**: every forward pass produces **exactly one new token per active sequence**.

Between any two decode steps, the model has just finished one iteration and is about to start the next. This creates a **natural, fine-grained scheduling boundary** after *every single token*.

At that exact moment, you can:

- Check which sequences in the current batch just generated an **end-of-sequence (EOS)** token or hit their max length → immediately remove them and free their KV cache slots.
- Check the waiting queue of new prompts → immediately prefill as many as will fit into the newly freed slots (prefill can be a smaller "ragged" sub-batch).
- Add the newly prefilled sequences to the active batch.

The batch membership is **recomputed every iteration** (hence "iteration-level scheduling"). The active batch size stays close to the hardware maximum almost continuously, because slots open up and get refilled *immediately*.

This is the core of **continuous batching**:

- Sequences **enter and leave the batch independently**.
- No more waiting for the entire batch to finish.
- The GPU stays busy decoding at near-max batch size nearly all the time.
- New user requests get served with much lower queuing delay.

In practice, many systems also support **chunked prefill** (breaking very long prompts into chunks that can be mixed with decode steps) and **paged KV cache** (vLLM's PagedAttention) to make memory allocation as flexible as virtual memory, further reducing fragmentation.

### 5\. How continuous batching relates to dynamic batching

The terminology in the field is **not perfectly standardized**, which causes some confusion:

- **Dynamic batching** is sometimes used broadly as an umbrella term for any batching that isn't purely static — i.e., batches whose composition can change based on runtime conditions.
- In many older or non-LLM contexts (e.g., image generation, where every output takes roughly the same time), "dynamic batching" means the request-level version I described in step 3: group on arrival/time window, but run the batch to completion.
- For LLMs specifically, **continuous batching** (or "dynamic continuous batching," "iteration-level batching") is the more advanced, token-granularity version. It is often described as a *form of* or *improvement upon* dynamic batching.

From the literature and production systems:

- The Orca paper (2022) introduced "continuous batching" via iteration-level scheduling.
- Many sources (including Anyscale's influential blog and vLLM docs) explicitly note that **continuous batching is also called dynamic batching** or **iteration-level scheduling**.
- Some articles distinguish them: "dynamic batching" for the coarser request-level grouping, and "continuous batching" for the finer per-iteration swapping that is ideal for LLMs (because outputs have highly variable lengths, unlike fixed-cost modalities like images).

In short:

- **Dynamic batching** (broader/weaker sense) = adaptively group requests rather than using rigid fixed batches.
- **Continuous batching** = the LLM-optimized version that operates at token/iteration granularity, allowing sequences to join/leave *continuously* between every decode step.

Virtually all high-performance LLM servers today implement the continuous/iteration-level version and may refer to it interchangeably as dynamic or continuous batching.

### 6\. Why the gains are so large (empirical impact)

Under real traffic with variable lengths, continuous batching commonly delivers **2–10× higher throughput** (sometimes up to 23× in early benchmarks) compared to static or simple dynamic request-level batching, while also reducing tail latencies. The GPU rarely has idle cycles waiting for a "batch boundary."

The pseudocode is almost identical to what I showed previously for dynamic batching, because in the LLM context they converge to the same idea:

```markdown
active_sequences = []  # with their KV caches

while True:
    # Decode one token for everything currently active (GPU fully utilized)
    if active_sequences:
        new_tokens = model.decode(active_sequences)  # batch size = current active count
    
    # Remove finished sequences immediately
    for seq in active_sequences.copy():
        if seq.is_finished(new_tokens[seq]):
            active_sequences.remove(seq)
            free_kv_cache(seq)
    
    # Prefill and add new requests into freed slots
    free_slots = max_batch_size - len(active_sequences)
    if free_slots > 0 and waiting_queue:
        new_prompts = waiting_queue.take_up_to(free_slots)
        prefilled = model.prefill(new_prompts)  # can be ragged/smaller
        active_sequences.extend(prefilled)
```

This loop runs continuously. The batch is never "frozen."

That's continuous batching from first principles. It directly exploits the iterative, token-by-token nature of autoregressive generation combined with KV caching to keep the expensive GPU compute saturated at all times.

If you'd like, I can go deeper into supporting techniques like PagedAttention, chunked prefill, or how scheduling policies (e.g., prioritizing shorter requests) interact with it.

so if one user submits a request that's 500 tokens and another submits a 1000 token request, is it guaranteed that all 500 tokens from request 1 will be sequence 1 and 1000 tokens from request 2 will be sequence 2? in other words, if a user submits a request that's like 10k tokens, will their request be broken up into multiple sequences that decode in parallel?

**No** — a single user request (one prompt, regardless of whether it’s 500 tokens, 1,000 tokens, or 10k tokens) is **always treated as one indivisible logical sequence** in continuous batching.

Here’s why, explained from first principles again, so the guarantee is crystal clear.

### 1\. What a “sequence” actually is in the inference engine

In the LLM serving runtime (vLLM, TGI, TensorRT-LLM, etc.), the active batch is a collection of **independent sequences**. Each sequence is defined by:

- Its own **KV cache** (the stored keys and values for every token generated so far for *that specific request*).
- Its own **current length** (how many tokens have been processed).
- Its own **next-token sampler** and stopping criteria (EOS, max tokens, etc.).

The transformer’s attention mechanism for any new token in sequence *i* must attend to **all previous tokens that belong to the same sequence** *i*. You cannot mix tokens from different user requests into the same sequence without breaking correctness.

So the engine fundamentally treats each user submission as **exactly one sequence**.

### 2\. Example with your two users

- User A submits a 500-token prompt → it becomes **sequence 1**.
- User B submits a 1,000-token prompt → it becomes **sequence 2**.

During prefill:

- The engine may prefill them together in one ragged batch (if both are waiting at the same time), or separately, depending on timing.
- After prefill, sequence 1 has a KV cache of length 500.
- Sequence 2 has a KV cache of length 1,000.

During every decode iteration, the active batch might contain both:

- Sequence 1 (length 500 + generated tokens so far)
- Sequence 2 (length 1,000 + generated tokens so far)

They decode **in parallel** in the same GPU forward pass, but they remain **separate entries** in the batch. The 500 tokens of User A never become part of User B’s sequence, and vice versa. The mapping is **guaranteed 1:1** between a user request and a sequence for its entire lifetime.

### 3\. What happens with a 10k-token prompt?

A single 10k-token request is **still one sequence**. It is **not** broken up into ten 1k-token sequences that decode in parallel.

Why it can’t be split:

- The model’s attention for token #10,001 must be able to attend to **all 10,000 previous tokens of that same user**. If you split it into multiple sequences, the model would lose the full context and produce nonsense.
- The sampler, stopping logic, and streaming response back to the user are all tied to the original request ID.

So the engine keeps it as **one long sequence** with a single KV cache (or, in vLLM, a chain of paged blocks that together form that one sequence).

### 4\. The only exception: chunked prefill (still one sequence)

For very long prompts (e.g., 10k+ tokens), many modern systems use **chunked prefill** (also called incremental prefill):

- Instead of pre-filling all 10k tokens in one giant forward pass (which would block the GPU for a long time), the engine splits the prompt into chunks (e.g., 2k tokens each).
- It processes one chunk at a time, **interleaving** those chunked prefill steps with decode steps from *other* active sequences.
- But even while doing this, the request is still tracked as **a single sequence**. The KV cache is simply extended chunk-by-chunk until the whole prompt is cached. The sequence ID never changes, and it never becomes multiple parallel sequences.

After the final chunk is prefilled, the sequence joins the normal decode phase exactly like any shorter sequence.

### 5\. Summary (the guarantee you asked for)

- **Yes**: All tokens from User 1’s 500-token request stay together as **sequence 1** for the entire lifetime of that request.
- **Yes**: All tokens from User 2’s 1,000-token request stay together as **sequence 2**.
- **No**: A 10k-token request is **never** split into multiple independent sequences that decode in parallel. It remains exactly one sequence, even if its prefill is internally chunked for efficiency.

This design is what makes continuous batching correct and simple: the scheduler only decides *when* to add or remove whole sequences from the active batch; it never splits a user’s context across sequences.

That’s why the “dynamic” part of continuous batching only affects *which sequences* are in the batch at any given iteration, never the internal structure of any one user’s request.

could you explain request coalescing from first principles and how it connects to continuous batching?

**Request coalescing** is a general systems technique for handling **duplicate or identical concurrent requests** efficiently by merging them into a single execution, then sharing the result with all waiters. It prevents redundant work and protects backend resources from "thundering herd" problems (many simultaneous requests hammering the same thing).

I'll explain it **from first principles**, then show how it connects (or doesn't directly connect) to **continuous batching** in LLM inference.

### 1\. First principles: Why coalescing exists

Imagine a system with a slow, expensive operation (database query, API call, heavy computation, or — in our case — an LLM forward pass).

- Multiple clients request the **exact same thing** at roughly the same time (e.g., "What is the capital of France?" or the same prompt + parameters).
- Without coalescing: Each request triggers its own independent execution → duplicate work, wasted CPU/GPU/network, higher latency for everyone, and potential overload (cache stampede).

**Coalescing** solves this by adding a thin coordination layer:

- When the first request for a particular "key" (the unique identifier of the work) arrives, it starts the expensive operation and marks it as "in flight."
- Any subsequent identical requests that arrive while it's in flight do **not** start new work. Instead, they wait (block or register a callback) for the in-flight operation to finish.
- When the operation completes, the result is delivered to **all** waiting requests at once.

This is also commonly called **single-flight**, **request deduplication**, or **coalesced execution**.

Classic non-LLM examples:

- Web caches / CDNs (Varnish, Bunny.net): Multiple users request the same uncached image → only one request goes to the origin server; others wait and get the response when it arrives.
- Caching libraries (e.g., singleflight in Go, or in-process maps of Futures/Promises): If 50 threads ask for the same cache key at the same moment and it misses, only one thread does the DB lookup; the other 49 await the shared future.

Key properties from first principles:

- It only helps when requests are **identical** (same input, same parameters, same context).
- It adds a small synchronization overhead (usually a per-key lock or map lookup).
- It dramatically reduces load when duplicates are common (e.g., popular queries, shared prompts in multi-agent systems, or repeated tool calls).

### 2\. Request coalescing in the context of LLM inference

In LLM serving, coalescing is **less central** than in traditional web serving because:

- Most user prompts are **unique** (different questions, different conversation histories, different system prompts).
- Even similar prompts often have different sampling parameters (temperature, top-p), stopping criteria, or streaming requirements, so they aren't truly identical.
- The dominant cost is the **per-sequence KV cache and autoregressive decoding**, which is inherently per-request.

However, coalescing **does appear** in a few targeted places in advanced LLM systems:

- **Identical prompt deduplication**: If multiple users submit the *exact same prompt* (with identical parameters) at the same time, the engine can run the prefill once, share the initial KV cache or first token, and fan out the result. Some batch inference pipelines or custom routers implement this.
- **In multi-agent or RAG systems**: When many agents issue the same tool call or retrieval query, coalescing avoids duplicate LLM calls or database hits.
- **Frontend / API layer**: Before even reaching the LLM engine, a proxy or load balancer can coalesce duplicate requests using a map keyed on (prompt hash + parameters).

In practice, you see mentions of "request coalescing" or "query coalescing" in MLPerf load generators or in scaling layers around LLM servers (routers, caches, or Ray Serve setups), but it is **not** a core part of the mainstream inference engines like vLLM or TGI.

### 3\. How request coalescing connects to continuous batching

They operate at **different layers** and solve orthogonal problems, but they can complement each other nicely.

- **Continuous batching** (iteration-level scheduling) is about **maximizing GPU utilization** for *different* requests:
	- It dynamically adds/removes *independent sequences* from the active batch after every decode step.
		- It handles **heterogeneous** workloads (different prompt lengths, different output lengths).
		- The goal is to keep the batch size close to the hardware maximum at all times by swapping finished sequences with waiting ones (including prefill of new prompts).
		- It assumes requests are mostly unique and focuses on scheduling and memory management (often with PagedAttention).
- **Request coalescing** is about **avoiding duplicate work** when requests are *identical*:
	- It sits upstream (in the API layer, router, or a thin wrapper around the engine).
		- It reduces the *number* of sequences that ever enter the continuous batching scheduler.
		- If 10 users ask the exact same question simultaneously, coalescing turns that into 1 sequence instead of 10 → the continuous batcher has fewer sequences to manage, and the GPU does far less redundant prefill/decode work.

**Combined effect** (when both are used):

1. Incoming requests hit the coalescing layer first → identical ones are merged into a single logical request (one sequence, shared result).
2. The deduplicated (or unique) requests then enter the waiting queue for the continuous batching engine.
3. The engine applies iteration-level scheduling: prefill when slots are free, decode in parallel with other unrelated sequences, remove when done, etc.

Result: You get the best of both worlds — fewer total sequences (from coalescing) + near-100% GPU utilization on the remaining sequences (from continuous batching).

In high-traffic scenarios with popular prompts (e.g., common questions in a customer support bot, repeated evaluations, or shared agent tool calls), coalescing can provide multiplicative gains on top of continuous batching.

### 4\. Simple mental model summary

- **Continuous batching**: "How do I keep the GPU busy when I have many *different* ongoing generations with varying lengths?" → Rebatching every iteration, swapping sequences in and out.
- **Request coalescing**: "What if several users asked for the *exact same* thing right now?" → Run it once, share the answer, don't create duplicate sequences.

They are friends, not the same thing. Continuous batching is the heavy lifter for general LLM serving throughput. Request coalescing is a lightweight but powerful optimization you layer on top when duplicate requests are likely.

If your use case involves many repeated or near-identical prompts (e.g., batch evaluation, multi-agent systems, or high-traffic chat with common queries), adding coalescing upstream can meaningfully reduce load before continuous batching even sees the requests.

Would you like me to sketch pseudocode for a simple coalescer that feeds into a continuous batching loop, or dive into when coalescing is most/least effective in LLM systems?

can you explain token parallelism from first principles and how it relates to the concepts above?

**Token parallelism** (sometimes called **sequence parallelism** or **context parallelism** in specific contexts) is a technique that exploits parallelism **along the sequence (token) dimension** of a transformer, rather than across multiple independent requests or by splitting model weights.

I'll explain it **from first principles**, building on the autoregressive nature of LLMs and the concepts we've already covered (continuous/dynamic batching, request coalescing).

### 1\. First principles: Where the sequential bottleneck comes from

Recall how an LLM works:

- **Attention** is the key operation: For a new token, its Query vector attends to **all previous Keys and Values** (from the prompt + previously generated tokens).
- In the **prefill phase** (processing the entire prompt), **all tokens in the prompt can be processed largely in parallel**. The model computes Q, K, V for every prompt token at once, and the attention matrix is fully computable because no token depends on future tokens.
- In the **decode phase**, generation is **strictly autoregressive**: Token $t+1$ depends on token $t$ , which depends on $t-1$ , and so on. You cannot compute token 5 without first knowing token 4. This creates a **serial dependency** — only one new token per sequence per forward pass in the basic case.

GPUs are excellent at **data parallelism** (doing the same operation on many independent items) and **tensor parallelism** (splitting matrix multiplications across GPUs). But the autoregressive chain limits how much you can parallelize *within one sequence*.

**Token parallelism** asks: *Can we break or reduce this serial chain to compute multiple tokens (or parts of a long sequence) more concurrently?*

There are two main flavors:

#### A. Intra-sequence token parallelism during prefill (most common meaning in serving)

For **very long prompts** (thousands to millions of tokens), even the prefill phase becomes expensive if done on a single device.

- Split the long sequence of tokens **across multiple GPUs** along the **sequence length dimension**.
- Each GPU holds a shard of the prompt tokens (e.g., GPU 1 gets tokens 1–2048, GPU 2 gets 2049–4096, etc.).
- For attention computation (the part that needs cross-token information), the GPUs communicate only the necessary Q/K/V shards (often using ring-all-reduce or more efficient ring-attention patterns like in Context Parallelism).
- This is also called **Context Parallelism (CP)** or **Sequence Parallelism (SP)** in papers and systems.

**Why it works from first principles**:

- In prefill, there are **no causal dependencies preventing parallelism** across the prompt tokens (masking takes care of "future" tokens).
- The heavy compute (matrix multiplies in attention and FFN) scales with sequence length, so sharding the tokens distributes both compute and memory (especially the large KV cache for long contexts).

This is orthogonal to **tensor parallelism (TP)**, which shards the *model weights* (e.g., splitting the hidden dimension of a linear layer across GPUs). Many systems combine TP + Context Parallelism for extreme long-context inference.

#### B. Parallel token generation during decode (speculative / non-autoregressive style)

A different but related idea: Instead of generating **one token at a time**, try to predict **multiple future tokens in parallel** within one or few forward passes.

- Techniques include **speculative decoding** (small draft model proposes several tokens; big model verifies them in parallel), **Jacobi decoding**, **Parallel Token Prediction (PTP)**, or consistency models.
- The model or a helper predicts several candidate tokens at once, then accepts as many as possible in a single verification step.
- This breaks the strict "one token per step" serial limit by turning decode into something closer to batch-parallel token production.

This flavor is sometimes explicitly called **token parallelism** in decoding contexts because it increases the number of tokens produced **per inference step**.

### 2\. How token parallelism relates to continuous (dynamic) batching

They operate at **different granularities** and are highly complementary:

- **Continuous batching (iteration-level scheduling)**:
	- Focuses on **inter-request parallelism**: Many *different* user sequences share the same GPU forward pass.
		- It dynamically adds/removes whole sequences (requests) between every decode step to keep the batch size high.
		- It treats each user request as **one indivisible sequence** (as we discussed earlier — a 10k-token prompt stays as one sequence).
		- Goal: Maximize GPU utilization across heterogeneous requests with variable lengths.
- **Token parallelism**:
	- Focuses on **intra-sequence parallelism**: Parallelizing work *inside* a single long sequence (mostly in prefill, or via speculative methods in decode).
		- For a single long request, it can shard its tokens across GPUs or compute multiple tokens concurrently.
		- It does **not** split one user request into multiple independent sequences (that would break causality and correctness). Instead, it distributes the *computation* for that one sequence.

**How they connect in practice**:

- Continuous batching runs on top of (or alongside) the parallelism strategy.
- A system can do **continuous batching + tensor parallelism + context/token parallelism**.
	- Example: Multiple short requests are batched together via continuous batching.
		- A few very long requests have their context sharded via token/context parallelism across additional GPUs.
		- The scheduler still decides which sequences (whole requests) participate in each iteration.
- Token parallelism helps especially with **long-context workloads** that would otherwise bottleneck prefill or exhaust KV cache memory on one device. It reduces per-GPU memory pressure and speeds up the expensive prefill for those heavy sequences, allowing continuous batching to keep the overall system busier.

**Request coalescing** has almost no direct relation — it’s about deduplicating *identical* requests upstream so fewer unique sequences ever reach the batching + parallelism layers. It reduces the load before token parallelism or continuous batching even starts.

### 3\. Simple mental model summary

- **Continuous / dynamic batching**: "Keep the GPU busy by mixing many *different* user requests in every step — swap them in/out dynamically."
- **Token parallelism (context/sequence parallelism)**: "For any *single* long request, split its tokens across GPUs or compute multiple tokens faster to handle extreme lengths efficiently."
- **Together**: You get high **throughput across users** (from continuous batching) + good **scalability for long individual contexts** (from token parallelism) + model-size scaling (from tensor/pipeline parallelism).

In modern high-performance serving systems (vLLM, TensorRT-LLM, etc.), you’ll often see combinations like:

- Tensor Parallelism (for large models)
- Continuous batching + PagedAttention (for high concurrency)
- Context/Token Parallelism or chunked prefill (for million-token contexts)

The end result is that even a single 10k+ token request no longer monopolizes the GPU for a huge prefill — its work can be parallelized internally while the system continues serving other users via continuous batching.

If you’d like, I can go deeper into Context Parallelism (with ring attention), how it differs from tensor parallelism, or examples of speculative decoding as token-level parallelism in decode. Just let me know which direction interests you!