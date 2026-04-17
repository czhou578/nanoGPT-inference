class Sequence:
    """
    Holds the state of a single request.
    """
    def __init__(self, seq_id: int, prompt_tokens: list):
        self.seq_id = seq_id
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = []
        self.status = "WAITING"  # States: WAITING, RUNNING, DONE

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

class Scheduler:
    """
    Decides which sequences to run, tracking memory capabilities.
    """
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.waiting = []
        self.running = []

    def add_sequence(self, seq_id: int, prompt_tokens: list) -> Sequence:
        seq = Sequence(seq_id, prompt_tokens)
        self.waiting.append(seq)
        return seq

    def step(self) -> list:
        """
        One iteration of continuous batching scheduling.
        Returns a list of sequences that are allowed to run in this step.
        """
        # 1. Try to start WAITING sequences if we have enough memory
        while self.waiting:
            seq = self.waiting[0]
            # Calculate how many blocks the prompt needs
            blocks_needed = (len(seq.prompt_tokens) + self.block_manager.block_size - 1) // self.block_manager.block_size
            
            # We enforce strict > blocks_needed to leave a buffer for existing runners
            # that might need a new block for their next generated token.
            if len(self.block_manager.free_blocks) > blocks_needed:
                self.waiting.pop(0)
                self.block_manager.allocate(seq.seq_id, len(seq.prompt_tokens))
                seq.status = "RUNNING"
                self.running.append(seq)
            else:
                break # Not enough memory for the next prompt. Stop scheduling.

        # 2. Check Memory for Decoding (Preemption)
        # If any running sequence needs a new block to append a token, but memory is full,
        # we must preempt (evict) the youngest sequence to make room.
        for seq in self.running:
            if seq.total_tokens % self.block_manager.block_size == 0:
                if not self.block_manager.free_blocks:
                    self._preempt_newest_sequence()
                    
        return self.running

    def _preempt_newest_sequence(self):
        """
        Evict the most recently started sequence to free up blocks.
        We use 'recomputation' preemption: we throw away its generated tokens
        and put it back in WAITING perfectly.
        """
        if not self.running:
            return
            
        # The newest sequence is at the end of the running list
        seq_to_preempt = self.running.pop(-1)
        self.block_manager.free(seq_to_preempt.seq_id)
        
        seq_to_preempt.status = "WAITING"
        seq_to_preempt.generated_tokens = [] # Discard progress
        
        # Put it at the VERY FRONT of waiting so it is prioritized next time
        self.waiting.insert(0, seq_to_preempt)

    def free_finished_sequence(self, seq: Sequence):
        """Clean up when a sequence hits EOS."""
        self.block_manager.free(seq.seq_id)
        seq.status = "DONE"
        self.running.remove(seq)
