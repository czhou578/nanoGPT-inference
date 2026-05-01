from settings import CONFIG


class Sequence:
    """Holds the state of a single inference request."""
    def __init__(self, seq_id: int, prompt_tokens: list):
        self.seq_id = seq_id
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = []
        self.status = "WAITING"   # WAITING | PREFILLING | RUNNING | DONE
        self.prefill_cursor = 0   # how many prompt tokens have been processed so far

    @property
    def total_tokens(self):
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def prefill_done(self):
        return self.prefill_cursor >= len(self.prompt_tokens)


class Scheduler:
    """
    Mixed-batch scheduler with chunked prefill.

    step() returns [(seq, chunk), ...]:
      - chunk is a non-empty token slice  → PREFILLING: warm up the KV-cache, no token emitted.
      - chunk is []                       → RUNNING:    decode one new token from the cache.
    """
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.waiting    = []   # not yet started
        self.prefilling = []   # prompt partially processed
        self.running    = []   # prompt fully cached, generating tokens

    def add_sequence(self, seq_id: int, prompt_tokens: list) -> "Sequence":
        seq = Sequence(seq_id, prompt_tokens)
        self.waiting.append(seq)
        return seq

    def step(self) -> list:
        bs = self.block_manager.block_size

        # 1. Admit waiting sequences if there's enough free memory.
        #    Strict '>' keeps one buffer block for running sequences that may need
        #    a new block on their very next decode step.
        while self.waiting:
            seq = self.waiting[0]
            blocks_needed = (len(seq.prompt_tokens) + bs - 1) // bs
            if len(self.block_manager.free_blocks) > blocks_needed:
                self.waiting.pop(0)
                self.block_manager.allocate(seq.seq_id, len(seq.prompt_tokens))
                seq.status = "PREFILLING"
                self.prefilling.append(seq)
            else:
                break  # memory full; stop admitting

        # 2. Preempt the newest running sequence if a decode step would need
        #    a new block but none are free (total_tokens divisible by block_size
        #    means the last block just filled up).
        for seq in self.running:
            if seq.total_tokens % bs == 0 and not self.block_manager.free_blocks:
                self._preempt()

        # 3. Build the mixed batch.
        #    Prefill gets at most prefill_chunk_size tokens this step (shared budget).
        #    Decode sequences always run regardless of budget.
        budget = CONFIG.prefill_chunk_size
        batch = []

        for seq in list(self.prefilling):  # list() copy so we can remove mid-loop
            if budget <= 0:
                break
            n = min(len(seq.prompt_tokens) - seq.prefill_cursor, budget)
            chunk = seq.prompt_tokens[seq.prefill_cursor : seq.prefill_cursor + n]
            seq.prefill_cursor += n
            budget -= n
            batch.append((seq, chunk))
            if seq.prefill_done:
                self.prefilling.remove(seq)
                seq.status = "RUNNING"
                self.running.append(seq)

        for seq in self.running:
            batch.append((seq, []))  # empty chunk = decode mode

        return batch

    def _preempt(self):
        """Evict the newest running sequence back to WAITING (recomputation preemption)."""
        if not self.running:
            return
        seq = self.running.pop()
        self.block_manager.free(seq.seq_id)

        seq.prefill_cursor = 0
        seq.generated_tokens = []
        seq.status = "WAITING"
        
        self.waiting.insert(0, seq)  # front of queue so it's prioritised next step

    def free_finished_sequence(self, seq: "Sequence"):
        """Release a sequence's blocks when it hits EOS."""
        self.block_manager.free(seq.seq_id)
        seq.status = "DONE"
        
        if seq in self.running:
            self.running.remove(seq)
