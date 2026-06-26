"""vocab.py - Character-level vocabulary construction and encode/decode utilities.

Builds a bidirectional character-to-integer mapping from raw text (sorted
alphabetically), and provides encode/decode helpers for converting between
strings and integer token ID lists.
"""

from typing import Dict, List, Tuple


class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build character-level stoi and itos mappings from raw text.

        Collects all unique characters, sorts them alphabetically, and assigns
        a unique integer to each one starting from 0.

        Args:
            text: The raw corpus string to derive the vocabulary from.

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: (stoi, itos) where stoi maps
            each character to its integer ID and itos is the reverse mapping.
        """
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        unique_chars = sorted(set(text))
        stoi = {char: i for i, char in enumerate(unique_chars)}
        itos = {i: char for char, i in stoi.items()}
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        """Convert a string to a list of integer token IDs using stoi.

        Args:
            text: Input string to encode.
            stoi: Character-to-integer mapping from build_vocab.

        Returns:
            List[int]: Integer ID for each character in text.
        """
        return [stoi[char] for char in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        """Convert a list of integer token IDs back to a string using itos.

        Args:
            ids: List of integer token IDs to decode.
            itos: Integer-to-character mapping from build_vocab.

        Returns:
            str: The reconstructed string.
        """
        return "".join([itos[num] for num in ids])
