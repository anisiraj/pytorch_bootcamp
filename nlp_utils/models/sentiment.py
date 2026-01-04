"""Sentiment classification model."""

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_utils.models.pooling import MaskedMeanPool
from nlp_utils.tokenization.protocol import TokenizerProtocol


class ModelInput(TypedDict):
    """Type definition for model input batch."""
    input_ids: torch.Tensor  # [batch_size, seq_len]
    mask: torch.Tensor       # [batch_size, seq_len]


class SentimentModel(nn.Module):
    """
    Two-layer sentiment classification model with masked mean pooling and dropout.

    Architecture:
        Embedding → MaskedMeanPool → Dropout → Hidden Projection (ReLU) → Dropout → Classifier

    Args:
        tokenizer: Any tokenizer implementing TokenizerProtocol (vocab_size, pad_token_id).
                   Compatible with SimpleTokenizer, HuggingFace tokenizers, etc.
        embedding_dim: Embedding dimension (default: 256)
        hidden_dim: Hidden layer dimension (default: 128)
        num_categories: Number of output categories (default: 2)
        dropout: Dropout probability (default: 0.3)

    Example:
        >>> from nlp_utils import SimpleTokenizer
        >>> tokenizer = SimpleTokenizer()
        >>> tokenizer.build_vocab(['hello world'])
        >>> model = SentimentModel(tokenizer, embedding_dim=256, hidden_dim=128, dropout=0.3)
        >>> batch = {'input_ids': torch.randint(0, 100, (32, 128)),
        ...          'mask': torch.ones(32, 128)}
        >>> logits = model(batch)  # [32, 2]
    """

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_categories: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_length = tokenizer.vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_index = tokenizer.pad_token_id
        self.num_categories = num_categories
        self.dropout_p = dropout

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_length,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_index
        )
        self.pool = MaskedMeanPool()
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.num_categories)

    def forward(self, batch: ModelInput) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: Dict with 'input_ids' and 'mask'

        Returns:
            logits: [batch, num_categories] - Raw logits for each category
        """
        x = batch['input_ids']    # [batch, seq_len]
        mask = batch['mask']      # [batch, seq_len]

        # 1. Embed tokens
        x = self.embedding(x)     # [batch, seq_len, emb_dim]

        # 2. Pool sequence to single vector
        x = self.pool(x, mask)    # [batch, emb_dim]

        # 3. Apply dropout after pooling
        x = self.dropout(x)       # [batch, emb_dim]

        # 4. Hidden projection
        x = self.fc_hidden(x)     # [batch, hidden_dim]
        x = F.relu(x)

        # 5. Apply dropout before classifier
        x = self.dropout(x)       # [batch, hidden_dim]

        # 6. Output projection
        x = self.classifier(x)    # [batch, num_categories]

        return x