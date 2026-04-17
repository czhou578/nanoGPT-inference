from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Language Model Dimensions
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    vocab_size: int = 1000
    max_seq_len: int = 512
    
    # Vision Model Dimensions
    vision_feature_length: int = 16  # Represents how many tokens an image outputs

    # Inference/System Settings
    block_size: int = 16  # Tokens per PagedAttention memory block

# Create a global configuration instance to be imported easily
CONFIG = ModelConfig()
