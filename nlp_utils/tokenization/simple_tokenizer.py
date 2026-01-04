"""Unified tokenizer with built-in vocabulary management."""

import re
from collections import Counter
from typing import Optional


class SimpleTokenizer:
    """
    Combined tokenization and vocabulary management.

    This class handles both tokenization (text → tokens) and vocabulary
    (tokens ↔ indices), providing a single cohesive interface similar to
    HuggingFace tokenizers.

    Args:
        lowercase: Convert text to lowercase before tokenization (default: True)
        remove_punctuation: Remove punctuation marks (default: True)
        min_length: Minimum token length to keep (default: 1)
        min_freq: Minimum frequency for token to be in vocabulary (default: 1)

    Example:
        >>> tokenizer = SimpleTokenizer()
        >>> tokenizer.build_vocab(['hello world', 'hello there'])
        >>> ids = tokenizer.encode('hello world')
        >>> tokens = tokenizer.decode(ids)
        >>> print(tokenizer.vocab_size)
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_length: int = 1,
        min_freq: int = 1
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_length = min_length
        self.min_freq = min_freq

        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        # Vocabulary mappings
        self.token2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2token = {0: self.pad_token, 1: self.unk_token}
        self.token_frequency = Counter()

        # Next available index (after special tokens)
        self._next_idx = 2

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into tokens.

        Args:
            text: Input text string

        Returns:
            List of tokens

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokens = tokenizer.tokenize("Hello, World!")
            >>> print(tokens)  # ['hello', 'world']
        """
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        tokens = [
            token for token in text.split()
            if len(token) >= self.min_length
        ]

        return tokens

    def build_vocab(self, texts: list[str]) -> 'SimpleTokenizer':
        """
        Build vocabulary from collection of texts.

        Args:
            texts: List of text strings

        Returns:
            self (for method chaining)

        Example:
            >>> tokenizer = SimpleTokenizer(min_freq=2)
            >>> tokenizer.build_vocab(['hello world', 'hello there', 'world peace'])
            >>> print(tokenizer.vocab_size)
        """
        # Count token frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.token_frequency.update(tokens)

        # Sort by frequency (most common first)
        sorted_tokens = sorted(
            self.token_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Add tokens to vocabulary
        for token, freq in sorted_tokens:
            if freq < self.min_freq:
                continue
            if token not in self.token2idx:
                self.token2idx[token] = self._next_idx
                self.idx2token[self._next_idx] = token
                self._next_idx += 1

        print(f"Vocabulary size: {len(self.token2idx)}")
        return self

    def encode(self, text: str) -> list[int]:
        """
        Tokenize and encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token indices

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokenizer.build_vocab(['hello world'])
            >>> ids = tokenizer.encode('hello world')
            >>> print(ids)  # [2, 3] (after PAD=0, UNK=1)
        """
        tokens = self.tokenize(text)
        return [self.token2idx.get(token, self.unk_token_id) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """
        Decode token IDs back to tokens.

        Args:
            indices: List of token indices

        Returns:
            List of tokens

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokenizer.build_vocab(['hello world'])
            >>> tokens = tokenizer.decode([2, 3])
            >>> print(tokens)  # ['hello', 'world']
        """
        return [self.idx2token.get(idx, self.unk_token) for idx in indices]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2idx)

    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size for embedding layer initialization.

        Returns:
            Total number of tokens in vocabulary

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokenizer.build_vocab(['hello world'])
            >>> embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim=128)
        """
        return len(self.token2idx)

    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID (always 0)."""
        return self.token2idx[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        """Get the unknown token ID (always 1)."""
        return self.token2idx[self.unk_token]

    def save(self, path: str):
        """
        Save tokenizer state to file.

        Args:
            path: Path to save tokenizer state

        Example:
            >>> tokenizer.save('tokenizer.pkl')
        """
        import pickle
        state = {
            'config': {
                'lowercase': self.lowercase,
                'remove_punctuation': self.remove_punctuation,
                'min_length': self.min_length,
                'min_freq': self.min_freq,
            },
            'vocab': {
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'token_frequency': self.token_frequency,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """
        Load tokenizer state from file.

        Args:
            path: Path to saved tokenizer state

        Returns:
            Loaded tokenizer instance

        Example:
            >>> tokenizer = SimpleTokenizer.load('tokenizer.pkl')
        """
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Create instance with saved config
        tokenizer = cls(**state['config'])

        # Restore vocabulary
        tokenizer.token2idx = state['vocab']['token2idx']
        tokenizer.idx2token = state['vocab']['idx2token']
        tokenizer.token_frequency = state['vocab']['token_frequency']
        tokenizer._next_idx = len(tokenizer.token2idx)

        return tokenizer

    def __repr__(self) -> str:
        """String representation of tokenizer."""
        return (
            f"SimpleTokenizer(vocab_size={self.vocab_size}, "
            f"lowercase={self.lowercase}, "
            f"remove_punctuation={self.remove_punctuation})"
        )