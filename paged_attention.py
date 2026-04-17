class BlockManager:
    """
    A minimalist memory allocator for PagedAttention KV-Cache.
    Physical blocks are simply represented by integers (their index in memory).
    """
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        
        # A pool of available block indices [0, 1, ..., num_blocks-1]
        self.free_blocks = list(range(num_blocks))
        
        # Maps sequence ID to a list of physical block indices
        self.block_tables = {}

    def allocate(self, seq_id: int, num_tokens: int):
        """Allocate physical blocks for a new sequence."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < blocks_needed:
            raise ValueError("Out of memory blocks!")
            
        # Pop 'blocks_needed' indices from the free pool
        allocated = [self.free_blocks.pop(0) for _ in range(blocks_needed)]
        self.block_tables[seq_id] = allocated

    def append_slot(self, seq_id: int, current_num_tokens: int):
        """Allocate a new block during generation if the last block is full."""
        # If the number of tokens is perfectly divisible by block size, 
        # it means the current block is completely full, so we need a new one.
        if current_num_tokens % self.block_size == 0:
            if not self.free_blocks:
                raise ValueError("Out of memory slots for appending!")
            self.block_tables[seq_id].append(self.free_blocks.pop(0))

    def free(self, seq_id: int):
        """Release all blocks used by a sequence back to the free pool."""
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)
