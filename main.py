from torch.nn.functional import layer_norm
from torch import nn
from file_opener import open_txt_file
import torch
from tokenizer import generate_vocab, encode_text, create_samples, decode_ids
from model import (
    create_causal_mask,
    MultiHeadAttention,
    create_pos_encoding,
    create_ffn,
    transformer_forward,
)
from generate import generate
import math

text_1 = open_txt_file("Data/LLM_TEXT.txt")

encode_vocab, decode_vocab = generate_vocab(text_1)
encoded_text = encode_text(text_1, encode_vocab)

SEQ_LEN = 5
X_ids, Y_ids = create_samples(encoded_text, SEQ_LEN)

vocab_size = len(encode_vocab)
embed_dim = 8

embedding_matrix = torch.randn(vocab_size, embed_dim) * 0.01
x = embedding_matrix[X_ids]
x = x.float()

attention_model = MultiHeadAttention(d_model=embed_dim, num_heads=2)

ffn = create_ffn(embed_dim)
layer_norm = nn.LayerNorm(embed_dim)
layer_norm_2 = nn.LayerNorm(embed_dim)
output_layer = nn.Linear(embed_dim, vocab_size)

logits, weights = transformer_forward(
    X_ids,
    embedding_matrix,
    attention_model,
    ffn,
    layer_norm,
    layer_norm_2,
    output_layer,
)

mask = create_causal_mask(x.shape[1], x.device)
mask = mask.unsqueeze(0).unsqueeze(0)

attention_output, weights = attention_model(x, mask=mask)
residual_output = attention_output + x
normalized_output = layer_norm(residual_output)

ffn_output = ffn(normalized_output)

residual_output_2 = ffn_output + normalized_output
final_output = layer_norm_2(residual_output_2)
print(final_output.shape)

#print(attention_output.shape)
#print(weights.shape)
#print("logits.shape", logits.shape)
#print("weights.shape", weights.shape)

def forward_wrapper(input_ids):
    return transformer_forward(
        input_ids,
        embedding_matrix,
        attention_model,
        ffn,
        layer_norm,
        layer_norm_2,
        output_layer,
    )

start_tokens = torch.tensor(encoded_text[:3], dtype=torch.long)

generated_ids = generate(
    start_tokens=start_tokens,
    max_new_tokens=2,
    transformer_forward_fn=forward_wrapper,
)

generated_text = decode_ids(generated_ids, decode_vocab)
print("generated ids:", generated_ids)
print("generated text:", generated_text)



