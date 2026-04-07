import torch
import torch.nn.functional as F

def generate(start_tokens, max_new_tokens, transformer_forward_fn): # generating from starting tokens to max tokens needed
    generated = start_tokens.clone() # creating a copy of starting tokens not to break original tensor

    for _ in range(max_new_tokens): #generation cycle
        input_ids = generated.unsqueeze(0) # adding batch size
        # unsqueeze(0) -> adding a new axis in 0 position: (5,) -> (1, 5)
        logits, weights = transformer_forward_fn(input_ids) # moving sequence through transformer_forward function
        last_logits = logits[0, -1]

        temperature = 1.5
        last_logits = last_logits / temperature # taking logits from last token
        """why last? cuz model from the last position thinks which token must be next after
        this sequence."""
        penalty = 1.2 # penalty power
        for token_id in set(generated.tolist()):
            last_logits[token_id] = last_logits[token_id] / penalty # reducing logit of used token
        probs = F.softmax(last_logits, dim=-1) # turning grades to probabilities

        next_token = torch.multinomial(probs, num_samples=1) # choosing token with higher probability
        # argmax - returns not prob itself, but index of that prob
        generated = torch.cat([generated, next_token], dim=0)
        # adding a new token in the end of already generated sequence
    return generated