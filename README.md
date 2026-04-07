Transformer block (from scratch)
This project is a custom implementation of a Transformer-like architecture written in python.
The goal of the project is to deeply understand how modern language models (LLM) work internally - from tokenization to attention and text generation

Features:
-custom tokenizer implementation
-Embedding layer
-scaled dot-product attention
-multi-head attention mechanism
-Feed Forward Network (FFN)
-Residual connections + layer normalization
-positional encoding
-logits generation
-basic text generation
-repetition penalty and Loss function

How it works:
1. text is loaded from folder and tokenized into integers
2. tokens are converted into embeddings
3. poitional encoding is added
4. data passes through attention layers
5. feef forward network refines representations
6. final layer produces logits
7. model generates next tokens step by step

Tech Stack:
-Python
-Pytorch (basic usage)
-Numpy

Future Improvements:
-add a training loop (backpropagation + loss)
-improve tokenizer (subword)
-add batching support
-optimize generation(temperature, top-k, top-p)
