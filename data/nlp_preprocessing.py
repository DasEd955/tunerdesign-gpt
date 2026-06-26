"""nlp_preprocessing.py - Vocabulary building and padded sequence encoding for NLP datasets.

Converts lists of positive and negative text samples into a padded integer tensor
suitable for use with an embedding layer. Words are assigned IDs starting at 1
(0 is reserved for padding) in sorted alphabetical order.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        """Encode and pad positive and negative sentences into a 2D integer tensor.

        Builds a vocabulary from all unique words across both lists, assigns integer
        IDs starting at 1 (sorted alphabetically), encodes each sentence as a list of
        word IDs, concatenates positive + negative, and pads shorter sequences with 0s
        using nn.utils.rnn.pad_sequence(tensors, batch_first=True).

        Args:
            positive: List of positive-class sentences.
            negative: List of negative-class sentences.

        Returns:
            TensorType[float]: Padded integer tensor of shape
            (len(positive) + len(negative), max_sentence_length).
        """
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive + negative

        vocab = sorted({word for sentence in combined for word in sentence.split()})
        word_to_id = {word: i + 1 for i, word in enumerate(vocab)}

        encoded = [torch.tensor([word_to_id[w] for w in s.split()]) for s in combined]

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)
