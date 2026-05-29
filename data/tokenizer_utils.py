from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = list()
        for num in numbers:
            str_num = str(num)
            tokens = self._greedy_tokenize(str_num, vocab)
            result.append(tokens)
        return result
    
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
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
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens, words = self._greedy_tokenize(text, vocab), text.split()
        return round(len(tokens) / len(words), 4)
