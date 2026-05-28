import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import heapq

# Copy all the code from nanogpt-prefix-caching.py
with open("nanogpt-prefix-caching.py", "r") as f:
    code = f.read()

# Append a small test harness
test_code = """
model.eval()

# Test Identical Requests
req1 = Request(id=0, prompt_tokens=encode("To be or not to be"), max_new_tokens=5)
req2 = Request(id=1, prompt_tokens=encode("To be or not to be"), max_new_tokens=5)

try:
    scheduler = scheduled_generate(model, [req1, req2], token_budget=8, max_kv_tokens=100)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
"""

with open("test_script.py", "w") as f:
    f.write(code + "\n" + test_code)

python3 test_script.py
