import numpy as np
from settings import CONFIG

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class TransformerBlock():
    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Randomly initialize dummy weights for Query, Key, Value, and Output projections
        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_v = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_o = np.random.randn(hidden_size, hidden_size) * 0.02
        
        # Feed-Forward Network projections
        self.W_up = np.random.randn(hidden_size, hidden_size * 4) * 0.02
        self.W_down = np.random.randn(hidden_size * 4, hidden_size) * 0.02

    def forward(self, x):
        """
        x shape: (seq_len, hidden_size)
        """
        seq_len = x.shape[0]
        
        # 1. Self-Attention
        Q = (x @ self.W_q).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = (x @ self.W_k).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = (x @ self.W_v).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        
        # Q @ K^T / sqrt(d)
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        
        # Causal Mask (prevent looking at future tokens)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores - mask * 1e9
        
        attn_weights = softmax(scores, axis=-1)
        
        # Attention Output
        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, self.hidden_size)
        
        # Residual + Projection
        x = x + (attn_output @ self.W_o)
        
        # 2. Feed-Forward Network with ReLU
        ffn_up = np.maximum(0, x @ self.W_up)
        ffn_output = ffn_up @ self.W_down
        
        # Residual
        x = x + ffn_output
        
        return x

class Dummy_LLM():
    def __init__(self):
        # Create a tiny embedding matrix and LM head
        self.embedding_matrix = np.random.randn(CONFIG.vocab_size, CONFIG.hidden_size) * 0.02
        self.lm_head = np.random.randn(CONFIG.hidden_size, CONFIG.vocab_size) * 0.02
        
        # Instantiate layers
        self.blocks = [
            TransformerBlock(CONFIG.hidden_size, CONFIG.num_heads) 
            for _ in range(CONFIG.num_layers)
        ]
    
    def generate(self, text, vision_features, num_tokens_to_generate=5):
        # 1. Tokenize & Embed the input text
        tokens = text.split()
        token_ids = [hash(token) % CONFIG.vocab_size for token in tokens]
        text_embeddings = self.embedding_matrix[token_ids]

        # 3. Combine Vision + Text embeddings
        if vision_features is not None:
            # Ensure proper shape before concatenation
            vision_features = np.array(vision_features).reshape(-1, CONFIG.hidden_size)
            embeddings = np.concatenate([vision_features, text_embeddings], axis=0)
        else:
            embeddings = text_embeddings
        
        # 4. Synthesize the generation loop (Sequence prediction)
        generated_tokens = []
        x = embeddings
        
        # Unroll the naive loop
        for _ in range(num_tokens_to_generate):
            # Forward pass through transformer layers
            hidden_states = x
            for block in self.blocks:
                hidden_states = block.forward(hidden_states)
            
            # Predict from the very last token
            last_hidden_state = hidden_states[-1, :]
            logits = last_hidden_state @ self.lm_head
            
            next_token_id = int(np.argmax(logits))
            generated_tokens.append(next_token_id)
            
            # Autoregressive step: Append the new token's embedding to input sequence
            next_emb = self.embedding_matrix[next_token_id].reshape(1, CONFIG.hidden_size)
            x = np.concatenate([x, next_emb], axis=0)
            
        return {
            "prompt_length": text_embeddings.shape[0],
            "vision_length": vision_features.shape[0] if vision_features is not None else 0,
            "generated_token_ids": generated_tokens
        }