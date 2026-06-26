"""tokenizer_utils.py - Greedy longest-match tokenization utilities.

Provides helpers for tokenizing text against a fixed vocabulary using a greedy
left-to-right longest match strategy, plus token count and fertility score metrics.
"""

from typing import List, Dict


class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        """Tokenize a list of integers into subword token lists using greedy longest match.

        Converts each integer to its string representation, then splits it into the
        fewest tokens possible by always choosing the longest matching vocab entry
        starting from the current position.

        Args:
            numbers: List of integers to tokenize.
            vocab: Dictionary mapping token strings to integer IDs.

        Returns:
            List[List[str]]: One token list per input number.
        """
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = list()
        for num in numbers:
            str_num = str(num)
            tokens = self._greedy_tokenize(str_num, vocab)
            result.append(tokens)
        return result
    
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        """Split text into tokens using greedy left-to-right longest vocab match.

        At each position tries decreasing substring lengths until a vocab match is
        found. Falls back to the single character if no match exists.

        Args:
            text: String to tokenize.
            vocab: Dictionary of known token strings.

        Returns:
            List[str]: Ordered list of tokens covering the full input.
        """
        # Greedy Algorithm tokenizer: move left to right through the text
        # Always choose the longest token from the vocab that matches
        
        tokens, i = list(), 0
        while i < len(text):
            best = None
            for length in range(len(text) - i, 0, -1):
                substr = text[i:i + length]
                if substr in vocab:
                    best = substr
                    break
            if best is None:
                tokens.append(text[i])
                i += 1
            else:
                tokens.append(best)
                i += len(best)
        return tokens

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        """Count the number of tokens produced by greedy tokenization of text.

        Args:
            text: Input string to tokenize.
            vocab: Dictionary of known token strings.

        Returns:
            int: Total number of tokens in the greedy tokenization of text.
        """
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        """Compute the fertility score (tokens per word ratio) for text.

        Higher fertility means the vocabulary fragments words into more pieces,
        which increases sequence length and model cost. Lower is more efficient.

        Args:
            text: Input string to evaluate.
            vocab: Dictionary of known token strings.

        Returns:
            float: Average number of tokens per word, rounded to 4 decimal places.
        """
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens, words = self._greedy_tokenize(text, vocab), text.split()
        return round(len(tokens) / len(words), 4)
