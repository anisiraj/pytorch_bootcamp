"""Classifier models built on transformer components."""

import torch
import torch.nn as nn

from nlp_utils.models.pooling import MaskedMeanPool
from nlp_utils.models.attention import (
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerEncoderBlock,
)


class MHAMeanPooledClassifier(nn.Module):
    """
    Classification head that applies masked mean pooling followed by a linear layer.

    Takes multi-head attention output [batch, seq_len, embedding_dim] and produces
    class logits [batch, num_target_classes].

    Args:
        embedding_dim: Dimension of input embeddings.
        num_target_classes: Number of output classes.

    Example:
        >>> classifier = MHAMeanPooledClassifier(embedding_dim=128, num_target_classes=2)
        >>> x = torch.randn(4, 20, 128)  # [batch, seq_len, embedding_dim]
        >>> mask = torch.ones(4, 20)
        >>> logits = classifier(x, mask=mask)  # [batch, 2]
    """

    def __init__(self, embedding_dim: int, num_target_classes: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_target_classes = num_target_classes
        self.mm_pool = MaskedMeanPool()
        self.classifier = nn.Linear(embedding_dim, self.num_target_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: pool then classify.

        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            mask: Optional padding mask [batch, seq_len]

        Returns:
            Logits tensor [batch, num_target_classes]
        """
        return self.classifier(self.mm_pool(x, mask=mask))


class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier.

    Architecture: Embedding -> Positional Encoding -> Transformer Encoder -> Mean Pool -> Linear

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of token embeddings.
        attention_heads: Number of attention heads.
        num_target_classes: Number of output classes.
        max_input_len: Maximum input sequence length for positional encoding.

    Example:
        >>> model = TransformerClassifier(vocab_size=5000, embedding_dim=128,
        ...                               attention_heads=4, num_target_classes=2)
        >>> input_ids = torch.randint(0, 5000, (4, 20))
        >>> mask = torch.ones(4, 20)
        >>> logits = model(input_ids, mask=mask)  # [batch, 2]
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 attention_heads: int = 4, num_target_classes: int = 2,
                 max_input_len: int = 512):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.num_target_classes = num_target_classes
        self.max_len = max_input_len

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Positional encoding
        self.pe = PositionalEncoding(self.embedding_dim, self.max_len)

        # Transformer encoder block
        mha = MultiHeadAttention(d_model=self.embedding_dim,
                                 num_heads=self.attention_heads)
        ff = FeedForward(d_model=self.embedding_dim)
        self.encoder_block = TransformerEncoderBlock(mha=mha, ff=ff)

        # Classification head
        self.classifier = MHAMeanPooledClassifier(
            embedding_dim=self.embedding_dim,
            num_target_classes=self.num_target_classes
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of transformer classifier.

        Args:
            input_ids: Token IDs [batch, seq_len]
            mask: Optional padding mask [batch, seq_len]

        Returns:
            Logits tensor [batch, num_target_classes]
        """
        # Embed tokens
        x = self.embedding(input_ids)  # [batch, seq_len, embedding_dim]

        # Add positional encoding
        x = self.pe(x)  # [batch, seq_len, embedding_dim]

        # Pass through transformer encoder
        x = self.encoder_block(x, mask=mask)  # [batch, seq_len, embedding_dim]

        # Pool and classify
        logits = self.classifier(x, mask=mask)  # [batch, num_target_classes]

        return logits
