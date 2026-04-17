import numpy as np
from settings import CONFIG


class Vision():
    def __init__(self):
        pass
    
    def encode(self, image_bytes):
        # Image sequence needs to match the hidden_size for concatenation
        vectors = np.random.rand(CONFIG.vision_feature_length, CONFIG.hidden_size)
        return vectors