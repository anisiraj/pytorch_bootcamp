"""Attention mechanisms for transformers."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor [batch, seq_len, d_k]
        K: Key tensor [batch, seq_len, d_k]
        V: Value tensor [batch, seq_len, d_k]
        mask: Optional mask [batch, seq_len] - 1 for real tokens, 0 for padding

    Returns:
        output: Attention output [batch, seq_len, d_k]
        attention: Attention weights [batch, seq_len, seq_len]
    """
    # Compute attention scores: Q @ K^T
    score = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]

    # Get dimension for scaling
    d_k = K.shape[-1]

    # Scale by sqrt(d_k)
    score = score / (d_k ** 0.5)

    # Apply mask if provided
    if mask is not None:
        # Expand mask from [batch, seq_len] to [batch, 1, seq_len]
        mask = mask.unsqueeze(1)
        # Set masked positions to -inf so softmax gives them 0 probability
        score = score.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get attention weights
    attention = F.softmax(score, dim=-1)

    # Apply attention to values
    output = attention @ V

    return output, attention


def create_padding_mask(seq, pad_token_id=0):
    """
    Create padding mask from input sequence.

    Args:
        seq: Input sequence [batch, seq_len]
        pad_token_id: ID used for padding tokens (default: 0)

    Returns:
        mask: Binary mask [batch, seq_len] - 1 for real tokens, 0 for padding
    """
    return torch.ones_like(seq).masked_fill(seq == pad_token_id, 0)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits the input into multiple heads, applies scaled dot-product attention
    to each head independently, then combines the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projection matrices (no bias for attention projections)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """
        Split embeddings into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Reshaped tensor [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, embedding_dim = x.shape
        assert embedding_dim == self.d_model, f"Expected d_model={self.d_model}, got {embedding_dim}"

        # Reshape and transpose
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine multiple heads back into single embedding.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Combined tensor [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert num_heads == self.num_heads, f"Expected {self.num_heads} heads, got {num_heads}"

        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        x = x.transpose(1, 2)

        # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, d_model]
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional padding mask [batch, seq_len]

        Returns:
            output: Attention output [batch, seq_len, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len, seq_len]
        """
        # 1. Project Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]

        # 2. Split into heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len, head_dim]
        K = self.split_heads(K)  # [batch, num_heads, seq_len, head_dim]
        V = self.split_heads(V)  # [batch, num_heads, seq_len, head_dim]

        # 3. Apply attention to each head
        # Need to handle mask expansion for multiple heads
        if mask is not None:
            # Expand mask: [batch, seq_len] -> [batch, 1, seq_len]
            # This will broadcast across all heads
            mask = mask.unsqueeze(1)

        # Compute attention scores for all heads at once
        scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            # Expand mask one more time for broadcasting across heads
            # [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        attention_output = attention_weights @ V  # [batch, num_heads, seq_len, head_dim]

        # 4. Combine heads
        combined = self.combine_heads(attention_output)  # [batch, seq_len, d_model]

        # 5. Final projection
        output = self.W_o(combined)  # [batch, seq_len, d_model]

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Add sinusoidal positional information to token embeddings.

    Uses sine and cosine functions of different frequencies to encode
    position information, allowing the model to learn relative positions.
    """

    @staticmethod
    def calculate_positional_encodings(d_model: int, max_len: int) -> torch.Tensor:
        """
        Calculate sinusoidal positional encodings.

        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length

        Returns:
            Positional encoding tensor [1, max_len, d_model]
        """
        # Create empty positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        # Create position indices [0, 1, 2, ..., max_len-1]
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate division term for different frequencies
        # div_term = 10000^(-2i/d_model) for i = 0, 1, 2, ...
        div_term = 10000 ** (-torch.arange(0, d_model, 2).float() / d_model)

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(positions * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Add batch dimension for broadcasting [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        return pe

    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length to pre-compute (default: 512)
        """
        super().__init__()
        self._d_model = d_model
        self._max_len = max_len
        self._register_encoding()

    def _register_encoding(self):
        """Register positional encodings as a buffer (not a parameter)."""
        pe = PositionalEncoding.calculate_positional_encodings(
            self._d_model,
            self._max_len
        )
        self.register_buffer('pe', pe)

    @property
    def d_model(self) -> int:
        """Get embedding dimension."""
        return self._d_model

    @property
    def max_len(self) -> int:
        """Get maximum sequence length."""
        return self._max_len

    @d_model.setter
    def d_model(self, value: int):
        """Set embedding dimension and recompute encodings."""
        self._d_model = value
        self._register_encoding()

    @max_len.setter
    def max_len(self, value: int):
        """Set maximum sequence length and recompute encodings."""
        self._max_len = value
        self._register_encoding()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]

        Raises:
            ValueError: If input dimensions don't match d_model or shape is wrong
        """
        seq_len, embedding_dim = x.size(-2), x.size(-1)

        # Validate embedding dimension
        if embedding_dim != self.d_model:
            raise ValueError(
                f"Input embedding dimension ({embedding_dim}) must match "
                f"d_model ({self.d_model})"
            )

        # Validate input shape
        if len(x.shape) != 3:
            raise ValueError(
                f"Input must be 3D [batch, seq_len, d_model], got shape {x.shape}"
            )

        # Add positional encoding (broadcasts across batch dimension)
        return x + self.pe[:, :seq_len, :]


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applies two linear transformations with ReLU activation and dropout.
    Standard transformer architecture uses 4x expansion in the hidden layer.
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout_rate: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension (embedding size)
            d_ff: Hidden layer dimension (default: 4 * d_model)
            dropout_rate: Dropout probability (default: 0.1)
        """
        super().__init__()

        # Default to 4x expansion if d_ff not specified
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Expansion: d_model -> d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)

        # Contraction: d_ff -> d_model
        self.contractor = nn.Sequential(
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Expansion with ReLU activation
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)

        # Contraction back to d_model
        x = self.contractor(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention and feed-forward layers.

    Architecture: Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    Uses residual connections and layer normalization.
    """

    def __init__(self, mha: MultiHeadAttention = None, ff: FeedForward = None):
        """
        Initialize transformer encoder block.

        Args:
            mha: Multi-head attention module
            ff: Feed-forward network module

        Raises:
            AssertionError: If mha or ff is None, or if d_model values don't match
        """
        super().__init__()

        # Validate that both components are provided
        assert mha is not None, "MultiHeadAttention (mha) cannot be None"
        assert ff is not None, "FeedForward (ff) cannot be None"

        # Validate that d_model is consistent across components
        assert mha.d_model == ff.d_model, \
            f"d_model mismatch: mha.d_model ({mha.d_model}) != ff.d_model ({ff.d_model})"

        self.ff = ff
        self.mha = mha

        # Get d_model from the MultiHeadAttention instance
        self.ln1 = nn.LayerNorm(mha.d_model)
        self.ln2 = nn.LayerNorm(mha.d_model)

    @property
    def d_model(self) -> int:
        """Get embedding dimension from the multi-head attention module."""
        return self.mha.d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of transformer encoder block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional padding mask [batch_size, seq_len] - 1 for real, 0 for padding

        Returns:
            Output tensor [batch_size, seq_len, d_model]

        Raises:
            AssertionError: If input embedding dimension doesn't match d_model
        """
        # Validate input embedding dimension matches d_model
        assert x.size(-1) == self.mha.d_model, \
            f"Input embedding dim ({x.size(-1)}) must match d_model ({self.mha.d_model})"

        # Multi-head attention with residual connection and layer norm
        attns, weights = self.mha(x, mask=mask)
        x = self.ln1(x + attns)

        # Feed-forward with residual connection and layer norm
        x = self.ln2(x + self.ff(x))

        return x
