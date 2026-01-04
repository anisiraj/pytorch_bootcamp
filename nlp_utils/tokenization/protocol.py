"""Protocol definition for tokenizer interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    Protocol defining the required interface for tokenizers.

    Any tokenizer (SimpleTokenizer, HuggingFace, custom) must implement
    these attributes and methods to be compatible with nlp_utils models.

    This allows models to work with any tokenizer implementation without
    tight coupling, following the Liskov Substitution Principle.

    Required Attributes:
        vocab_size: Total number of tokens in vocabulary
        pad_token_id: ID of the padding token

    Required Methods:
        encode(text): Convert text to list of token IDs
        decode(ids): Convert token IDs to list of tokens

    Example:
        >>> # SimpleTokenizer implements this protocol
        >>> from nlp_utils import SimpleTokenizer
        >>> tokenizer = SimpleTokenizer()
        >>> isinstance(tokenizer, TokenizerProtocol)  # True

        >>> # HuggingFace tokenizers also work
        >>> from transformers import AutoTokenizer
        >>> hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> # Works because it has vocab_size, pad_token_id, encode, decode
    """

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in vocabulary."""
        ...

    @property
    def pad_token_id(self) -> int:
        """ID of the padding token."""
        ...

    def encode(self, text: str) -> list[int]:
        """
        Encode text to list of token IDs.

        Args:
            text: Input text string

        Returns:
            List of token indices
        """
        ...

    def decode(self, ids: list[int]) -> list[str]:
        """
        Decode token IDs to list of tokens.

        Args:
            ids: List of token indices

        Returns:
            List of tokens
        """
        ...