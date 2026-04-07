import math
import torch
import torch.nn.functional as F
from torch import nn

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scaled_scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scaled_scores = scaled_scores + mask

        weights = F.softmax(scaled_scores, dim=-1)

        attention_output = weights @ V

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        # print("attention_output after concat: ", attention_output.shape)

        output = self.W_O(attention_output)
        # print("final output after W_O: ", output.shape)

        return output, weights


def create_pos_encoding(seq_len, d_model, device):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    return pos_encoding.unsqueeze(0)

def create_ffn(d_model):
    return nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.ReLU(),
        nn.Linear(d_model * 4, d_model),
    )

def transformer_forward(
        input_ids,
        embedding_matrix,
        attention_model,
        layer_norm,
        ffn,
        layer_norm_2,
        output_layer
):
    x = embedding_matrix[input_ids]
    x = x.float()

    seq_len = x.shape[1]
    d_model = x.shape[2]

    pos_encoding = create_pos_encoding(seq_len, d_model, x.device)
    x = x + pos_encoding

    mask = create_causal_mask(seq_len, x.device)
    mask = mask.unsqueeze(0).unsqueeze(0)

    attention_output, weights = attention_model(x, mask=mask)
    residual_output = attention_output + x
    normalized_output = layer_norm(residual_output)

    ffn_output = ffn(normalized_output)
    residual_output_2= ffn_output + normalized_output
    final_output = layer_norm_2(residual_output_2)

    logits = output_layer(final_output)
    return logits, weights
