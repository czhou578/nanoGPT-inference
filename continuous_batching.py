import numpy as np
from settings import CONFIG
from paged_attention import BlockManager
from scheduler import Scheduler, Sequence


class Engine:
    """
    Continuous Batching Engine with Chunked Prefill.

    Every step, the scheduler returns a mixed batch of:
      - (seq, chunk): sequences still being prefilled, with a slice of prompt tokens to process.
      - (seq, []):    sequences in decode mode, each generating exactly one new token.

    Chunked prefill prevents a long new prompt from stalling the decoding of existing sequences:
    instead of blocking the GPU for one giant prefill pass, the prompt is spread over many
    smaller steps interleaved with normal decode steps.
    """
    def __init__(self, llm):
        self.llm = llm
        self.block_manager = BlockManager(CONFIG.block_size, CONFIG.num_gpu_blocks)
        self.scheduler = Scheduler(self.block_manager)
        self._next_seq_id = 0

    def submit(self, text: str, vision_features=None) -> int:
        """
        Accepts a new text request and places it in the WAITING queue.
        Returns the sequence ID used to retrieve the result later.
        """
        token_ids = [hash(w) % CONFIG.vocab_size for w in text.split()]
        if vision_features is not None:
            n = np.array(vision_features).shape[0]
            # Vision tokens prepended as out-of-vocab placeholder IDs
            token_ids = list(range(CONFIG.vocab_size, CONFIG.vocab_size + n)) + token_ids
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.scheduler.add_sequence(seq_id, token_ids)
        return seq_id

    def _forward_prefill_chunk(self, seq: Sequence, chunk: list[int]) -> None:
        """
        Process a chunk of prompt tokens WITHOUT generating a new output token.

        During prefill we are computing and (conceptually) storing the Key/Value
        vectors for the chunk tokens into the KV-cache.  No new token is emitted —
        we just "warm up" the cache so that future decode steps can attend to them.

        Why no output token?
        If the prompt has 128 tokens and chunk_size=32, we don't want to generate
        a new token after every 32-token chunk — we haven't seen the full prompt yet.
        We only generate once the entire prompt has been processed (is_prefill_done).
        """
        # Embed only the chunk tokens (not the whole prompt).
        # The KV-cache already holds embeddings for tokens 0..prefill_cursor-chunk.
        chunk_ids = [t % CONFIG.vocab_size for t in chunk]
        x = self.llm.embedding_matrix[chunk_ids]

        # Run the forward pass so KV vectors are computed and cached (conceptually).
        for block in self.llm.blocks:
            x = block.forward(x)
        return x  # shape: (len(token_ids), hidden_size)

    def run(self, max_new_tokens: int = 10, eos_token_id: int = 0) -> tuple[dict, list]:
        """Run the loop until all sequences are done. Returns (results, trace)."""
        results, trace = {}, []

        for step in range(max_new_tokens):
            batch = self.scheduler.step()
            if not batch and not (self.scheduler.waiting or self.scheduler.prefilling or self.scheduler.running):
                break

            finished = []
            for seq, chunk in batch:
                if chunk:
                    # ── PREFILL CHUNK ────────────────────────────────────────────
                    # Run the chunk through the model to build the KV-cache.
                    # No token is emitted — we haven't seen the full prompt yet.
                    self._forward(chunk)
                    # Tell the block manager which slots were just consumed.
                    start = seq.prefill_cursor - len(chunk)
                    for i in range(len(chunk)):
                        self.block_manager.append_slot(seq.seq_id, start + i)
                    new_token_id = None

                else:
                    # ── DECODE ───────────────────────────────────────────────────
                    # Prompt is fully cached. Run all tokens to get the next one.
                    x = self._forward(seq.prompt_tokens + seq.generated_tokens)
                    new_token_id = int(np.argmax(x[-1] @ self.llm.lm_head))
                    seq.generated_tokens.append(new_token_id)
                    self.block_manager.append_slot(seq.seq_id, seq.total_tokens - 1)
                    if new_token_id == eos_token_id or seq.total_tokens >= CONFIG.max_seq_len:
                        finished.append(seq)

                trace.append({
                    "seq_id":         seq.seq_id,
                    "step":           step,
                    "phase":          "PREFILL_CHUNK" if chunk else "DECODE",
                    "prefill_cursor": seq.prefill_cursor,
                    "prompt_length":  len(seq.prompt_tokens),
                    "chunk_size":     len(chunk),
                    "num_generated":  len(seq.generated_tokens),
                    "total_tokens":   seq.total_tokens,
                    "block_table":    list(self.block_manager.block_tables.get(seq.seq_id, [])),
                    "free_blocks":    len(self.block_manager.free_blocks),
                    "new_token_id":   new_token_id,
                })

            for seq in finished:
                results[seq.seq_id] = seq.generated_tokens
                self.scheduler.free_finished_sequence(seq)

        # Collect sequences that hit max_new_tokens without EOS
        for seq in self.scheduler.running:
            results[seq.seq_id] = seq.generated_tokens

        return results, trace
