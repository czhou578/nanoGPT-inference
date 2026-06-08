/**
 * simulationData.js
 * 
 * Central data for all simulation modes:
 * - Mode metadata (descriptions, icons)
 * - Pre-defined simulation scenarios
 * - Color palettes for request/token visualization
 */

// ── Mode Definitions ──────────────────────────────────────────────────────────

export const MODES = [
  {
    id: 'kv-cache',
    label: 'KV Cache',
    icon: '🗂️',
    shortDesc: 'Cache key-value pairs to avoid redundant computation during autoregressive decoding.',
    description:
      'In autoregressive generation, each new token depends on all previous tokens. Without caching, the model recomputes keys and values for every prior token at every step — O(n²) work. The KV cache stores previously computed K and V tensors so only the new token needs a forward pass, reducing decode cost to O(n) per step.',
  },
  {
    id: 'continuous-batching',
    label: 'Continuous Batching',
    icon: '📦',
    shortDesc: 'Dynamically add/remove requests from the GPU batch at the token level.',
    description:
      'Static batching wastes GPU cycles: when one request finishes early, its slot sits idle until the entire batch completes. Continuous batching fills empty slots immediately with new requests, maximizing GPU utilization. Each request independently tracks its own sequence length and KV cache.',
  },
  {
    id: 'chunked-prefill',
    label: 'Chunked Prefill',
    icon: '✂️',
    shortDesc: 'Split long prompts into chunks and interleave them with decode steps.',
    description:
      'LLM inference has two phases: Prefill (processing the input prompt — compute-heavy) and Decode (generating tokens one-by-one — memory-bandwidth bound). Chunked Prefill breaks long prompts into smaller blocks so prefill computation can be interleaved alongside decode work, preventing long prompts from stalling active generation.',
  },
  {
    id: 'paged-attention',
    label: 'PagedAttention',
    icon: '📄',
    shortDesc: 'Manage KV cache memory with virtual pages, like an OS manages RAM.',
    description:
      'Inspired by OS virtual memory, PagedAttention partitions the KV cache into fixed-size blocks (pages) that can be stored non-contiguously in GPU memory. A block table maps logical positions to physical blocks. This eliminates fragmentation, avoids pre-allocating for max sequence length, and enables many more concurrent requests.',
  },
  {
    id: 'scheduling',
    label: 'Scheduling',
    icon: '📋',
    shortDesc: 'Intelligently allocate GPU resources and preempt lower-priority requests.',
    description:
      'The scheduler decides which requests to run, when to admit new ones, and when to preempt (pause and evict) active requests under memory pressure. It supports FCFS and priority policies, ensuring high-priority requests get served first while preventing starvation of lower-priority work.',
  },
  {
    id: 'speculative-decoding',
    label: 'Speculative Decoding',
    icon: '🎯',
    shortDesc: 'Use a fast draft model to propose tokens, then verify them in parallel.',
    description:
      'A small, fast "draft" model proposes K candidate tokens. The large "target" model then verifies all K candidates in a single forward pass. Accepted tokens skip expensive individual decode steps. When a candidate is rejected, the target model\'s distribution is used to resample. This can generate multiple tokens per forward pass of the target model.',
  },
];

// ── Color Palette for Requests ────────────────────────────────────────────────

export const REQUEST_COLORS = [
  { bg: 'rgba(99, 102, 241, 0.25)', border: '#6366f1', text: '#a5b4fc', label: 'Indigo' },
  { bg: 'rgba(236, 72, 153, 0.25)', border: '#ec4899', text: '#f9a8d4', label: 'Pink' },
  { bg: 'rgba(34, 197, 94, 0.25)',  border: '#22c55e', text: '#86efac', label: 'Green' },
  { bg: 'rgba(249, 115, 22, 0.25)', border: '#f97316', text: '#fdba74', label: 'Orange' },
  { bg: 'rgba(14, 165, 233, 0.25)', border: '#0ea5e9', text: '#7dd3fc', label: 'Sky' },
  { bg: 'rgba(168, 85, 247, 0.25)', border: '#a855f7', text: '#d8b4fe', label: 'Purple' },
  { bg: 'rgba(234, 179, 8, 0.25)',  border: '#eab308', text: '#fde047', label: 'Yellow' },
  { bg: 'rgba(20, 184, 166, 0.25)', border: '#14b8a6', text: '#5eead4', label: 'Teal' },
];

// ── KV Cache Simulation Data ──────────────────────────────────────────────────

export const KV_CACHE_DATA = {
  promptTokens: ['The', 'quick', 'brown', 'fox', 'jumps'],
  generatedTokens: ['over', 'the', 'lazy', 'dog', '.', 'It', 'was', 'a'],
  maxSteps: 8,
};

// ── Continuous Batching Simulation Data ───────────────────────────────────────

export const CONTINUOUS_BATCHING_DATA = {
  maxBatchSize: 4,
  requests: [
    { id: 0, prompt: 'Tell me a joke', arrivalStep: 0, maxNewTokens: 6 },
    { id: 1, prompt: 'What is AI?',    arrivalStep: 0, maxNewTokens: 8 },
    { id: 2, prompt: 'Hello world',    arrivalStep: 1, maxNewTokens: 5 },
    { id: 3, prompt: 'Explain GPUs',   arrivalStep: 2, maxNewTokens: 7 },
    { id: 4, prompt: 'Write code',     arrivalStep: 4, maxNewTokens: 6 },
    { id: 5, prompt: 'Summarize',      arrivalStep: 5, maxNewTokens: 4 },
    { id: 6, prompt: 'Translate',      arrivalStep: 7, maxNewTokens: 5 },
    { id: 7, prompt: 'Debug this',     arrivalStep: 9, maxNewTokens: 3 },
  ],
};

// ── Chunked Prefill Simulation Data ──────────────────────────────────────────

export const CHUNKED_PREFILL_DATA = {
  tokenBudget: 8,
  requests: [
    { id: 0, promptLength: 4,  maxNewTokens: 5, arrivalStep: 0 },
    { id: 1, promptLength: 12, maxNewTokens: 4, arrivalStep: 1 },
    { id: 2, promptLength: 6,  maxNewTokens: 3, arrivalStep: 4 },
  ],
};

// ── PagedAttention Simulation Data ───────────────────────────────────────────

export const PAGED_ATTENTION_DATA = {
  blockSize: 4,
  totalPhysicalBlocks: 16,
  requests: [
    { id: 0, promptLength: 6,  maxNewTokens: 6 },
    { id: 1, promptLength: 3,  maxNewTokens: 5 },
    { id: 2, promptLength: 5,  maxNewTokens: 4 },
  ],
};

// ── Scheduling Simulation Data ───────────────────────────────────────────────

export const SCHEDULING_DATA = {
  maxBatchSize: 3,
  maxKvTokens: 24,
  requests: [
    { id: 0, promptLength: 4, maxNewTokens: 6, priority: 2, arrivalStep: 0, label: 'Low' },
    { id: 1, promptLength: 3, maxNewTokens: 5, priority: 1, arrivalStep: 0, label: 'Med' },
    { id: 2, promptLength: 5, maxNewTokens: 4, priority: 2, arrivalStep: 1, label: 'Low' },
    { id: 3, promptLength: 3, maxNewTokens: 5, priority: 0, arrivalStep: 3, label: 'High' },
    { id: 4, promptLength: 4, maxNewTokens: 3, priority: 1, arrivalStep: 5, label: 'Med' },
  ],
};

// ── Speculative Decoding Simulation Data ─────────────────────────────────────

export const SPECULATIVE_DECODING_DATA = {
  K: 4, // draft tokens per round
  rounds: [
    { draftTokens: ['over', 'the', 'lazy', 'dog'],   accepted: [true, true, true, true],  bonus: '.' },
    { draftTokens: ['It',   'ran',  'fast', 'away'],  accepted: [true, true, false],        resample: 'quickly' },
    { draftTokens: ['and',  'hid',  'under', 'the'],  accepted: [true, true, true, false],  resample: 'behind' },
    { draftTokens: ['a',    'big',  'old',   'tree'],  accepted: [true, true, true, true],  bonus: '.' },
  ],
};
