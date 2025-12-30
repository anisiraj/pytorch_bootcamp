"""Vocabulary class for text encoding/decoding."""

from collections import Counter
from typing import Callable


type Tokenizer = Callable[[str], list[str]]


class Vocabulary:
    """
    Vocabulary for text tokenization and encoding.

    Args:
        tokenizer: Function that converts text to list of tokens
        min_freq: Minimum frequency for a token to be included in vocabulary
    """

    def __init__(self, tokenizer: Tokenizer, min_freq=1):
        self.token2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2token = {0: '<PAD>', 1: '<UNK>'}
        self.token_frequency = Counter()
        self.tokenize = tokenizer
        # 0,1 are reserved for special tokens
        self.start_token_index = 2
        self.min_freq = min_freq

    def _update_token_entry(self, idx, token):
        """Add token to vocabulary mappings."""
        self.token2idx[token] = idx
        self.idx2token[idx] = token

    def build_from_texts(self, texts):
        """
        Build vocabulary from a collection of texts.

        Args:
            texts: Iterable of text strings

        Returns:
            self (for method chaining)
        """
        for text in texts:
            tokens = self.tokenize(text)
            self.token_frequency.update(tokens)

        sorted_tokens = sorted(
            self.token_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        idx = self.start_token_index
        for token, freq in sorted_tokens:
            if freq < self.min_freq:
                continue
            self._update_token_entry(idx=idx, token=token)
            idx += 1

        print(f"Vocabulary size: {len(self.token2idx)}")
        return self

    def decode(self, indices: list[int]):
        """
        Convert list of indices to list of tokens.

        Args:
            indices: List of token indices

        Returns:
            List of tokens
        """
        return [self.idx2token.get(idx, '<UNK>') for idx in indices]

    def encode(self, text: str):
        """
        Convert text to list of token indices.

        Args:
            text: Input text string

        Returns:
            List of token indices
        """
        tokens = self.tokenize(text)
        unk_idx = self.token2idx.get('<UNK>')
        return [self.token2idx.get(token, unk_idx) for token in tokens]

    def __len__(self):
        """Return vocabulary size."""
        return len(self.token2idx)