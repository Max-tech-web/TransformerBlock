import re
import torch

UNK_TOKEN = "<unk>"

def generate_vocab(text_1):
    chunks = re.split(r'(\w+|[^\w\s])', text_1)
    tokens = [c for c in chunks if c.strip()]
    unique_tokens = list(dict.fromkeys(tokens))

    if UNK_TOKEN not in unique_tokens:
        unique_tokens.append(UNK_TOKEN)

    encode_vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    decode_vocab = {idx: token for idx, token in enumerate(unique_tokens)}
    return encode_vocab, decode_vocab



def encode_text(text_1, encode_vocab):
    tokens = re.split(r'(\w+|[^\w\s])', text_1)
    tokens = [t for t in tokens if t.strip()]

    unk_id = encode_vocab[UNK_TOKEN]
    token_ids = [encode_vocab.get(token, unk_id) for token in tokens]
    return token_ids


def create_samples(token_ids, seq_len):
    X = []
    Y = []

    for i in range(len(token_ids) - seq_len):
        x_seq = token_ids[i : i + seq_len]
        y_seq = token_ids[i + 1 : i + seq_len + 1]
        X.append(x_seq)
        Y.append(y_seq)
    return torch.tensor(X), torch.tensor(Y)

def decode_ids(token_ids, decode_vocab):
    return " ".join(decode_vocab[idx.item()] for idx in token_ids)