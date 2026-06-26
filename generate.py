"""generate.py - Autoregressive text generation for the character-level GPT model.

Samples new characters one at a time by repeatedly running the model, taking
the last-position logits, converting to a probability distribution via softmax,
and sampling the next token. Uses a fixed Generator seed for reproducibility.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        """Autoregressively generate new_chars characters from a seed context.

        At each step:
            1. Crops context to context_length if it exceeds the model's window.
            2. Runs model(context) and takes the last position's logits.
            3. Converts logits to probabilities with softmax(dim=-1).
            4. Samples the next token with torch.multinomial (seeded generator).
            5. Appends the sampled token to context with torch.cat.
            6. Maps the token integer to a character via int_to_char and accumulates.

        Args:
            model: Trained GPT model that returns logits of shape (1, T, vocab_size).
            new_chars: Number of new characters to generate.
            context: Integer token-ID tensor of shape (1, T) used as the initial seed.
            context_length: Maximum context window the model supports.
            int_to_char: Mapping from integer token ID to character string.

        Returns:
            str: The generated text as a single string of length new_chars.
        """
        generator = torch.Generator()
        generator.manual_seed(0)
        result = list()
        for i in range(new_chars):
            # Crop context to max length the model can handle
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            
            # Forward Pass -> logits for every position
            logits = model(context)             # (1, T, vocab_size)
            last_logits = logits[:, -1, :]      # (1, vocab_size)
            probs = nn.functional.softmax(last_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, 1, generator=generator)

            # Append token to context & decode
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])
        
        return "".join(result)