"""Tests for pooling layers."""

import pytest
import torch
from nlp_utils.models.pooling import MaskedMeanPool


class TestMaskedMeanPool:
    """Test cases for MaskedMeanPool layer."""

    def test_basic_pooling(self):
        """Test basic pooling with no padding."""
        pool = MaskedMeanPool()

        # Create simple embeddings: batch_size=2, seq_len=3, emb_dim=4
        embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [2.0, 4.0, 6.0, 8.0],
             [3.0, 6.0, 9.0, 12.0]],
            [[0.0, 1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
        ])

        # All tokens are real (no padding)
        mask = torch.ones(2, 3)

        pooled = pool(embeddings, mask)

        # Expected: mean over sequence dimension
        expected = torch.tensor([
            [2.0, 4.0, 6.0, 8.0],  # mean of [1,2,3], [2,4,6], [3,6,9], [4,8,12]
            [1/3, 1/3, 1/3, 1/3]   # mean of [0,1,0], [1,0,0], [0,1,0], [1,0,0]
        ])

        assert pooled.shape == (2, 4), f"Expected shape (2, 4), got {pooled.shape}"
        assert torch.allclose(pooled, expected, atol=1e-6), \
            f"Pooling mismatch.\nGot:\n{pooled}\nExpected:\n{expected}"

    def test_with_padding(self):
        """Test pooling with padding tokens."""
        pool = MaskedMeanPool()

        # Create embeddings with padding
        embeddings = torch.tensor([
            [[1.0, 1.0],
             [2.0, 2.0],
             [99.0, 99.0]],  # This should be ignored (padding)
            [[3.0, 3.0],
             [99.0, 99.0],  # This should be ignored
             [99.0, 99.0]]  # This should be ignored
        ])

        # Mask: 1 for real tokens, 0 for padding
        mask = torch.tensor([
            [1.0, 1.0, 0.0],  # First two are real
            [1.0, 0.0, 0.0]   # Only first is real
        ])

        pooled = pool(embeddings, mask)

        # Expected: mean of only non-padded tokens
        expected = torch.tensor([
            [1.5, 1.5],  # mean of [1,2]
            [3.0, 3.0]   # only [3]
        ])

        assert pooled.shape == (2, 2), f"Expected shape (2, 2), got {pooled.shape}"
        assert torch.allclose(pooled, expected, atol=1e-6), \
            f"Pooling with padding failed.\nGot:\n{pooled}\nExpected:\n{expected}"

    def test_all_padding(self):
        """Test edge case where all tokens are padding."""
        pool = MaskedMeanPool()

        embeddings = torch.randn(2, 5, 10)
        mask = torch.zeros(2, 5)  # All padding

        pooled = pool(embeddings, mask)

        # Should return zero vectors (division by max(1, 0) = 1)
        assert pooled.shape == (2, 10)
        assert torch.allclose(pooled, torch.zeros(2, 10), atol=1e-6), \
            "All-padding case should produce zero vectors"

    def test_single_token(self):
        """Test with single token sequences."""
        pool = MaskedMeanPool()

        embeddings = torch.tensor([
            [[5.0, 10.0, 15.0]],
            [[2.0, 4.0, 6.0]]
        ])

        mask = torch.ones(2, 1)

        pooled = pool(embeddings, mask)

        # With single token, pooling should return the token itself
        expected = torch.tensor([
            [5.0, 10.0, 15.0],
            [2.0, 4.0, 6.0]
        ])

        assert pooled.shape == (2, 3)
        assert torch.allclose(pooled, expected, atol=1e-6)

    def test_different_sequence_lengths(self):
        """Test with varying sequence lengths via masking."""
        pool = MaskedMeanPool()

        batch_size = 4
        max_seq_len = 10
        emb_dim = 8

        embeddings = torch.randn(batch_size, max_seq_len, emb_dim)

        # Different sequence lengths: [3, 5, 1, 10]
        mask = torch.zeros(batch_size, max_seq_len)
        mask[0, :3] = 1
        mask[1, :5] = 1
        mask[2, :1] = 1
        mask[3, :] = 1

        pooled = pool(embeddings, mask)

        assert pooled.shape == (batch_size, emb_dim)

        # Verify manually for first sample
        expected_first = embeddings[0, :3, :].mean(dim=0)
        assert torch.allclose(pooled[0], expected_first, atol=1e-6)

    def test_backward_pass(self):
        """Test that gradients flow correctly through pooling."""
        pool = MaskedMeanPool()

        embeddings = torch.randn(2, 4, 6, requires_grad=True)
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ])

        pooled = pool(embeddings, mask)
        loss = pooled.sum()
        loss.backward()

        assert embeddings.grad is not None, "Gradients should flow to embeddings"

        # Check that padding positions have zero gradient
        assert torch.allclose(embeddings.grad[0, 3, :], torch.zeros(6), atol=1e-6), \
            "Padding positions should have zero gradient"
        assert torch.allclose(embeddings.grad[1, 2:, :], torch.zeros(2, 6), atol=1e-6), \
            "Padding positions should have zero gradient"

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        pool = MaskedMeanPool()

        embeddings = torch.randn(1, 5, 12)
        mask = torch.ones(1, 5)

        pooled = pool(embeddings, mask)

        assert pooled.shape == (1, 12)
        expected = embeddings[0].mean(dim=0).unsqueeze(0)
        assert torch.allclose(pooled, expected, atol=1e-6)

    def test_large_embedding_dim(self):
        """Test with large embedding dimensions."""
        pool = MaskedMeanPool()

        batch_size = 16
        seq_len = 128
        emb_dim = 768  # BERT-like dimension

        embeddings = torch.randn(batch_size, seq_len, emb_dim)
        mask = torch.ones(batch_size, seq_len)

        pooled = pool(embeddings, mask)

        assert pooled.shape == (batch_size, emb_dim)

    def test_float_mask(self):
        """Test with float mask values (soft masking)."""
        pool = MaskedMeanPool()

        embeddings = torch.ones(1, 3, 2)  # All ones for simplicity

        # Soft mask with weights
        mask = torch.tensor([[0.5, 1.0, 0.0]])

        pooled = pool(embeddings, mask)

        # With all-ones embeddings and weights [0.5, 1.0, 0.0]
        # Sum = 1.5, Count = 1.5, Mean = 1.0
        expected = torch.ones(1, 2)

        assert torch.allclose(pooled, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
