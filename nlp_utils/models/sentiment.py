"""Sentiment classification model."""

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_utils.models.pooling import MaskedMeanPool


class ModelInput(TypedDict):
    """Type definition for model input batch."""
    input_ids: torch.Tensor  # [batch_size, seq_len]
    mask: torch.Tensor       # [batch_size, seq_len]


class SentimentModel(nn.Module):
    """
    Two-layer sentiment classification model with masked mean pooling.

    Architecture:
        Embedding → MaskedMeanPool → Hidden Projection (ReLU) → Classifier

    Args:
        vocab: Vocabulary object with get_vocab_size() and pad_token_id
        embedding_dim: Embedding dimension (default: 256)
        hidden_dim: Hidden layer dimension (default: 128)
        num_categories: Number of output categories (default: 2)

    Example:
        >>> from nlp_utils import Vocabulary, tokenize
        >>> vocab = Vocabulary(tokenizer=tokenize)
        >>> vocab.build_from_texts(['hello world'])
        >>> model = SentimentModel(vocab, embedding_dim=256, hidden_dim=128)
        >>> batch = {'input_ids': torch.randint(0, 100, (32, 128)),
        ...          'mask': torch.ones(32, 128)}
        >>> logits = model(batch)  # [32, 2]
    """

    def __init__(
        self,
        vocab,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_categories: int = 2
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_length = len(self.vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_index = self.vocab.pad_token_id
        self.num_categories = num_categories

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_length,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_index
        )
        self.pool = MaskedMeanPool()
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

        # 3. Hidden projection
        x = self.fc_hidden(x)     # [batch, hidden_dim]
        x = F.relu(x)

        # 4. Output projection
        x = self.classifier(x)    # [batch, num_categories]

        return x