import numpy as np
from transformers import AutoProcessor

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
action_data = np.random.rand(4, 1000, 14)    # one batch of action chunks
tokens = tokenizer(action_data)              # tokens = list[int]
decoded_actions = tokenizer.decode(tokens)
print(decoded_actions.shape)

# action embedding (action2vec: Skip-gram or CBOW)