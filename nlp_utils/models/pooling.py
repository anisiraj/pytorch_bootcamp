"""Pooling layers for sequence models."""

import torch
import torch.nn as nn


class MaskedMeanPool(nn.Module):
    """
    Performs masked mean pooling over the sequence dimension.

    Takes embeddings and a mask, zeros out padding positions,
    and computes the mean of real (non-padded) tokens.

    Example:
        >>> pool = MaskedMeanPool()
        >>> embeddings = torch.randn(32, 128, 256)  # [batch, seq_len, emb_dim]
        >>> mask = torch.ones(32, 128)  # [batch, seq_len]
        >>> pooled = pool(embeddings, mask)  # [batch, emb_dim]
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pool embeddings using mask.

        Args:
            embeddings: [batch, seq_len, emb_dim] - Token embeddings
            mask: [batch, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            pooled: [batch, emb_dim] - Averaged embeddings
        """
        # Expand mask to match embedding dimensions
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]

        # Zero out padding positions
        masked_embeddings = embeddings * mask_expanded  # [batch, seq_len, emb_dim]

        # Sum over sequence length
        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, emb_dim]

        # Count real tokens (avoid division by zero)
        token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]

        # Compute mean
        pooled = sum_embeddings / token_counts  # [batch, emb_dim]

        return pooled