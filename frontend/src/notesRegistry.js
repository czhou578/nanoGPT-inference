/**
 * notesRegistry.js
 *
 * Central data module that:
 * 1. Glob-imports all ../notes/*.md files as raw strings at build time
 * 2. Defines the topic taxonomy and maps files to categories
 * 3. Extracts metadata (title, sources) from each file
 * 4. Exports helper functions for components to consume
 */

// Glob import all markdown files from notes/ as raw strings
const mdModules = import.meta.glob('../../notes/*.md', { query: '?raw', import: 'default', eager: true });

// ── Category definitions (flat list, ordered per user spec) ──────────────────

export const categories = [
  { id: 'streaming-generation', name: 'Streaming Generation', icon: '🌊' },
  { id: 'pipeline-parallelism', name: 'Pipeline Parallelism', icon: '🔗' },
  { id: 'prefetch-pipelines', name: 'Prefetch Pipelines', icon: '⚡' },
  { id: 'cuda-graphs', name: 'CUDA Graphs', icon: '📊' },
  { id: 'speculative-decoding', name: 'Speculative Decoding', icon: '🎯' },
  { id: 'paged-attention', name: 'PagedAttention', icon: '📄' },
  { id: 'kv-cache-quantization', name: 'KV Cache & Quantization', icon: '🗜️' },
  { id: 'dynamic-batching', name: 'Dynamic & Continuous Batching', icon: '📦' },
  { id: 'memory-offload', name: 'Memory Offload', icon: '💾' },
  { id: 'gpu-cpu-overlap', name: 'GPU–CPU Overlap', icon: '🔄' },
  { id: 'request-coalescing', name: 'Request Coalescing', icon: '🔀' },
  { id: 'disaggregated-prefill', name: 'Disaggregated Prefill/Decode', icon: '✂️' },
  { id: 'latency-metrics', name: 'Inference Latency Metrics', icon: '⏱️' },
  { id: 'hardware-fundamentals', name: 'Hardware Fundamentals', icon: '🖥️' },
  { id: 'roofline-model', name: 'Roofline Model & Arithmetic Intensity', icon: '📐' },
  { id: 'vllm-internals', name: 'vLLM Internals', icon: '⚙️' },
  { id: 'investor-framework', name: 'Investor Framework', icon: '💰' },
  { id: 'api-pricing', name: 'API Pricing & Inference Economics', icon: '💲' },
  // Placeholder categories (no files yet)
  { id: 'token-parallelism', name: 'Token Parallelism', icon: '🧩', placeholder: true },
  { id: 'fp8-kernels', name: 'FP8 Kernels', icon: '🔢', placeholder: true },
  { id: 'async-prefill', name: 'Asynchronous Prefill', icon: '🚀', placeholder: true },
  { id: 'early-exit-heads', name: 'Early Exit Heads', icon: '🚪', placeholder: true },
  { id: 'context-window-streaming', name: 'Context Window Streaming', icon: '🪟', placeholder: true },
];

// ── File → Category mapping ──────────────────────────────────────────────────

const fileMapping = {
  'Streaming Generation in LLM Inference Explained.md': 'streaming-generation',
  'pipeline-parallelism.md': 'pipeline-parallelism',
  'prefetch-pipelines.md': 'prefetch-pipelines',
  'CUDA Graphs_ First Principles and Importance.md': 'cuda-graphs',
  'speculative-decoding.md': 'speculative-decoding',
  'PagedAttention_ Efficient KV Cache Management.md': 'paged-attention',
  'quantization.md': 'kv-cache-quantization',
  'sliding-window-kv-eviction.md': 'kv-cache-quantization',
  'Dynamic Batching in LLM Inference Explained.md': 'dynamic-batching',
  'client-side-batching.md': 'dynamic-batching',
  'memory-offload.md': 'memory-offload',
  'gpu-cpu-overlap.md': 'gpu-cpu-overlap',
  'request-coalescing.md': 'request-coalescing',
  'disaggregated-prefill-decode.md': 'disaggregated-prefill',
  'LLM Inference Latency_ TTFT, ITL, E2E Explained.md': 'latency-metrics',
  'HBM vs SRAM_ Memory Fundamentals Compared.md': 'hardware-fundamentals',
  'reiner-pope-podcast.md': 'roofline-model',
  'vllm_concepts.md': 'vllm-internals',
  'investor.md': 'investor-framework',
  'api-pricing-inference-costs.md': 'api-pricing',
};

// Files to exclude from the knowledge base
const excludedFiles = new Set(['prompts.md']);

// ── Metadata extraction ─────────────────────────────────────────────────────

function extractTitle(content) {
  // Try YAML frontmatter title
  const frontmatterMatch = content.match(/^---\s*\n[\s\S]*?title:\s*"([^"]+)"[\s\S]*?---/);
  if (frontmatterMatch) return frontmatterMatch[1];

  // Try first # heading
  const headingMatch = content.match(/^#\s+(.+)$/m);
  if (headingMatch) return headingMatch[1];

  return 'Untitled';
}

function extractSources(content) {
  const sources = [];

  // YAML frontmatter source
  const frontmatterSource = content.match(/^---\s*\n[\s\S]*?source:\s*"([^"]+)"[\s\S]*?---/);
  if (frontmatterSource) {
    sources.push({ url: frontmatterSource[1], type: 'primary' });
  }

  // Inline "Source:" lines
  const sourceLines = content.match(/^Source:\s*(.+)$/gm);
  if (sourceLines) {
    sourceLines.forEach(line => {
      const text = line.replace(/^Source:\s*/, '');
      sources.push({ text, type: 'primary' });
    });
  }

  // Paper references (anything with "et al." or year in parens)
  const paperRefs = content.match(/(?:[A-Z][a-z]+\s+et\s+al\.\s*[,(]\s*\d{4}\s*\)?|[""]\s*[^""]+[""]\s*\(\d{4}\))/g);
  if (paperRefs) {
    paperRefs.forEach(ref => {
      if (!sources.some(s => s.text === ref)) {
        sources.push({ text: ref, type: 'paper' });
      }
    });
  }

  // URLs in the content
  const urlMatches = content.match(/https?:\/\/[^\s)>\]]+/g);
  if (urlMatches) {
    urlMatches.forEach(url => {
      if (!sources.some(s => s.url === url)) {
        sources.push({ url, type: 'link' });
      }
    });
  }

  return sources;
}

function stripFrontmatter(content) {
  return content.replace(/^---\s*\n[\s\S]*?---\s*\n/, '');
}

function slugify(filename) {
  return filename
    .replace(/\.md$/, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
}

// ── Build the notes registry ─────────────────────────────────────────────────

const notes = [];

for (const [path, rawContent] of Object.entries(mdModules)) {
  const filename = path.split('/').pop();

  if (excludedFiles.has(filename)) continue;

  const categoryId = fileMapping[filename];
  if (!categoryId) continue; // skip unmapped files

  const title = extractTitle(rawContent);
  const sources = extractSources(rawContent);
  const content = stripFrontmatter(rawContent);
  const slug = slugify(filename);

  notes.push({
    slug,
    filename,
    title,
    categoryId,
    content,
    sources,
  });
}

// ── Public API ───────────────────────────────────────────────────────────────

export function getNotes() {
  return notes;
}

export function getNoteBySlug(slug) {
  return notes.find(n => n.slug === slug) || null;
}

export function getNotesByCategory(categoryId) {
  return notes.filter(n => n.categoryId === categoryId);
}

export function getCategoriesWithNotes() {
  return categories.map(cat => ({
    ...cat,
    notes: getNotesByCategory(cat.id),
  }));
}

export function getAllSources() {
  const allSources = [];
  for (const note of notes) {
    for (const source of note.sources) {
      allSources.push({
        ...source,
        noteSlug: note.slug,
        noteTitle: note.title,
      });
    }
  }
  return allSources;
}
