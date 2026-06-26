"""dataset.py - Word-level batch loader for raw text datasets.

Tokenizes a raw string on whitespace and samples random (input, target) sequence
pairs for language model training. Uses torch.manual_seed(0) for reproducibility.
"""

import torch
from typing import List, Tuple


class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        """Sample a batch of (input, target) word-token sequences from raw text.

        Tokenizes by splitting on whitespace, then generates batch_size random
        start indices in [0, len(tokens) - context_length). For each index i:
            X = tokens[i : i + context_length]   (Input)
            Y = tokens[i+1 : i+1 + context_length]  (Target, shifted by 1)

        Args:
            raw_dataset: The full training corpus as a whitespace-delimited string.
            context_length: Number of word tokens in each sequence.
            batch_size: Number of (X, Y) pairs to sample.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: (X, Y) where each element is a
            list of batch_size sequences, each sequence being a list of word strings.
        """
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        # X -> Input; Y -> Target
        torch.manual_seed(0)
        data = raw_dataset.split()
        indices = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,)).tolist()
        X, Y = list(), list()
        for i in indices:
            X.append(data[i:i + context_length])
            Y.append(data[i + 1:i + 1 + context_length])
        return (X, Y)