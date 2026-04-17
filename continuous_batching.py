import numpy as np
from settings import CONFIG
from paged_attention import BlockManager
from scheduler import Scheduler

class Engine:
    """
    Continuous Batching Engine.

    Instead of a naive 'generate-one-request-to-completion' loop, this engine
    runs step-by-step. Every step, all RUNNING sequences emit exactly one token
    in parallel. New requests can join or leave the batch at any step boundary.
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
            num_vision_tokens = np.array(vision_features).shape[0]
            # Vision tokens use out-of-range IDs as placeholders
            vision_ids = list(range(CONFIG.vocab_size, CONFIG.vocab_size + num_vision_tokens))
            token_ids = vision_ids + token_ids

        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.scheduler.add_sequence(seq_id, token_ids)
        return seq_id

    def _forward_one_token(self, seq):
        """
        Run one decode step for a single sequence.
        Embeds all tokens so far, runs the transformer, returns the next token id.
        """
        all_token_ids = [t % CONFIG.vocab_size for t in seq.prompt_tokens + seq.generated_tokens]
        x = self.llm.embedding_matrix[all_token_ids]

        for block in self.llm.blocks:
            x = block.forward(x)

        logits = x[-1] @ self.llm.lm_head
        return int(np.argmax(logits))

    def run(self, max_new_tokens: int = 10, eos_token_id: int = 0) -> tuple[dict, list]:
        """
        Run the continuous batching loop until all sequences are DONE.

        Returns:
          results: dict mapping seq_id -> list of generated token ids
          trace:   list of per-step snapshots for visualization
        """
        results = {}
        trace = []

        for step in range(max_new_tokens):
            running = self.scheduler.step()

            if not running and not self.scheduler.waiting:
                break

            finished = []
            step_entries = []

            for seq in running:
                next_token = self._forward_one_token(seq)
                seq.generated_tokens.append(next_token)

                # Inform the block manager a slot was consumed
                self.block_manager.append_slot(seq.seq_id, seq.total_tokens - 1)

                # --- Trace snapshot for this sequence at this step ---
                step_entries.append({
                    "seq_id":              seq.seq_id,
                    "step":                step,
                    "scheduler_status":    seq.status,
                    "num_prompt_tokens":   len(seq.prompt_tokens),
                    "num_generated":       len(seq.generated_tokens),
                    "total_tokens":        seq.total_tokens,
                    "block_table":         list(self.block_manager.block_tables.get(seq.seq_id, [])),
                    "free_blocks":         len(self.block_manager.free_blocks),
                    "new_token_id":        next_token,
                })

                if next_token == eos_token_id or seq.total_tokens >= CONFIG.max_seq_len:
                    finished.append(seq)

            trace.extend(step_entries)

            for seq in finished:
                results[seq.seq_id] = seq.generated_tokens
                self.scheduler.free_finished_sequence(seq)

        # Collect any sequences that hit max_new_tokens without EOS
        for seq in self.scheduler.running:
            results[seq.seq_id] = seq.generated_tokens

        return results, trace
