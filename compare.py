import torch
from transformers import GPT2Model
from models.gpt2 import GPT2Model as MyGPT2Model
from modules.attention import CausalSelfAttention
from utils import model_size_to_params

# Load OpenAI's GPT-2 model
openai_gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# Load Your GPT-2 Model
my_gpt2 = MyGPT2Model.from_pretrained(model='gpt2', **model_size_to_params('gpt2'))

# Generate Sample Inputs
input_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                          [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], 
                         [1, 1, 1, 1, 1, 1, 1, 1]])

# Forward Pass on OpenAI GPT-2
openai_outputs = openai_gpt2(input_ids=input_ids, attention_mask=att_mask, output_attentions=True)
openai_attention_output = openai_outputs.attentions[0]  # First layer's attention output

# Extract Self-Attention Module from Your Model
my_self_attention: CausalSelfAttention = my_gpt2.gpt_layers[0].self_attention

# Generate Hidden States for Testing
hidden_states = torch.rand(2, 8, 768)  # Batch size 2, seq_len 8, hidden dim 768
attention_mask = torch.ones(2, 1, 1, 8)  # Example attention mask

# Compute Key, Query, Value for Your Model
key_layer = my_self_attention.transform(hidden_states, my_self_attention.key)
query_layer = my_self_attention.transform(hidden_states, my_self_attention.query)
value_layer = my_self_attention.transform(hidden_states, my_self_attention.value)

# Compute Your Attention Output
your_attention_output = my_self_attention.attention(key_layer, query_layer, value_layer, attention_mask)

# Print Shapes
print("OpenAI Attention Output Shape:", openai_attention_output.shape)  # Expected: [batch_size, num_heads, seq_len, seq_len]
print("Your Attention Output Shape:", your_attention_output.shape)  # Should match OpenAI’s shape

# Compare Attention Outputs
print(your_attention_output)
print(openai_attention_output)
assert torch.allclose(your_attention_output, openai_attention_output, atol=1e-1, rtol=1e-1), "Mismatch in Attention Outputs!"
print("✅ Your attention function matches OpenAI GPT-2!")

# Optional: Print a few values for debugging
print("Your Attention Output (Sample):\n", your_attention_output[0, :, :, :3])
print("OpenAI Attention Output (Sample):\n", openai_attention_output[0, :, :, :3])
