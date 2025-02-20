import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    bs, _, _, seq_l = attention_mask.shape
    causal_mask = torch.triu(torch.ones(seq_l, seq_l, device=attention_mask.device), diagonal=1) # (sl ,sl)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, -10000).unsqueeze(0).unsqueeze(0) #(1, 1, sl, sl)

    sqrt_dk = key.shape[-1] ** 0.5
    att_score = torch.matmul(query, key.transpose(-2, -1))/sqrt_dk # Q bs, nh, sl, hiddensize-> bs, nh, sl, sl
    att_score = att_score + causal_mask

    # att_score (bs,_, sl, sl)
    if attention_mask is not None: 
      att_score = att_score + attention_mask # (bs,1,1,sl)
    att_score = nn.functional.softmax(att_score.float(), dim = -1, dtype=torch.float32) # bs, nh, sl, sl
    att_score = self.dropout(att_score)
    attention_output = torch.matmul(att_score, value) # ns, nh, sl, hs
    return attention_output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
