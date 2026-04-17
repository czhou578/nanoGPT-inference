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
        Returns the sequence ID that can be used to retrieve the result later.
        """
        # Tokenize the prompt to a flat list of token ids
        token_ids = [hash(w) % CONFIG.vocab_size for w in text.split()]

        # Prepend vision tokens (if any) as dummy integer tokens
        if vision_features is not None:
            num_vision_tokens = np.array(vision_features).shape[0]
            # Use out-of-range IDs (> vocab_size) as placeholders for vision tokens
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
        # Clamp token ids to the vocab for embedding lookup (vision tokens are clamped)
        all_token_ids = [t % CONFIG.vocab_size for t in seq.prompt_tokens + seq.generated_tokens]
        x = self.llm.embedding_matrix[all_token_ids]

        for block in self.llm.blocks:
            x = block.forward(x)

        # Predict from the last position
        logits = x[-1] @ self.llm.lm_head
        return int(np.argmax(logits))

    def run(self, max_new_tokens: int = 10, eos_token_id: int = 0) -> dict:
        """
        Run the continuous batching loop until all sequences are DONE.
        Returns a dict mapping seq_id -> list_of_generated_token_ids.
        """
        results = {}

        for _step in range(max_new_tokens):
            # Scheduler decides which sequences run this step
            running = self.scheduler.step()

            if not running and not self.scheduler.waiting:
                break  # Everything is finished

            # --- Parallel decode: every running sequence emits exactly one token ---
            finished = []
            for seq in running:
                next_token = self._forward_one_token(seq)
                seq.generated_tokens.append(next_token)

                # Inform the block manager a slot was consumed
                self.block_manager.append_slot(seq.seq_id, seq.total_tokens - 1)

                if next_token == eos_token_id or seq.total_tokens >= CONFIG.max_seq_len:
                    finished.append(seq)

            # Free finished sequences at the end of the step (not mid-loop)
            for seq in finished:
                results[seq.seq_id] = seq.generated_tokens
                self.scheduler.free_finished_sequence(seq)

        # Collect any sequences that hit max_new_tokens without an EOS
        for seq in self.scheduler.running:
            results[seq.seq_id] = seq.generated_tokens

        return results
