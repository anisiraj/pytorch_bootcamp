"""Tests for attention mechanisms."""

import pytest
import torch
import torch.nn as nn
from nlp_utils.models.attention import (
    scaled_dot_product_attention,
    create_padding_mask,
    MultiHeadAttention,
)


class TestScaledDotProductAttention:
    """Test cases for scaled dot-product attention."""

    def test_no_mask(self):
        """Test basic attention without masking."""
        Q = K = V = torch.randn(2, 10, 64)
        output, weights = scaled_dot_product_attention(Q, K, V)

        # Check shapes
        assert output.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {output.shape}"
        assert weights.shape == (2, 10, 10), f"Expected shape (2, 10, 10), got {weights.shape}"

        # Attention weights should sum to 1 along last dimension
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10)), \
            "Attention weights should sum to 1"

    def test_with_mask(self):
        """Test attention with padding mask."""
        Q = K = V = torch.randn(2, 10, 64)

        # Create mask with different sequence lengths
        mask = torch.ones(2, 10)
        mask[0, 7:] = 0  # First sequence: 7 real tokens, 3 padding
        mask[1, 5:] = 0  # Second sequence: 5 real tokens, 5 padding

        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Check shapes
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 10, 10)

        # Attention weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10))

        # Verify masked positions get ~0 attention
        assert weights[0, :7, 7:].max() < 1e-6, \
            "Padding positions should not receive attention!"
        assert weights[1, :5, 5:].max() < 1e-6, \
            "Padding positions should not receive attention!"

    def test_all_masked(self):
        """Test edge case where all tokens are masked."""
        Q = K = V = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5)  # All masked

        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # With all positions masked, attention weights become NaN after softmax(-inf)
        # This is expected behavior - in practice you wouldn't mask everything
        assert torch.isnan(weights).all(), \
            "All-masked attention should produce NaN (expected behavior)"

    def test_single_token(self):
        """Test with single token sequences."""
        Q = K = V = torch.randn(3, 1, 16)
        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (3, 1, 16)
        assert weights.shape == (3, 1, 1)
        assert torch.allclose(weights, torch.ones(3, 1, 1)), \
            "Single token should attend to itself with weight 1"

    def test_scaling_effect(self):
        """Test that scaling prevents large values."""
        d_k = 64
        Q = K = V = torch.randn(1, 5, d_k)

        # Without scaling (for comparison)
        unscaled_scores = Q @ K.transpose(-2, -1)

        # With scaling
        output, weights = scaled_dot_product_attention(Q, K, V)
        scaled_scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

        # Scaled scores should have smaller variance
        assert unscaled_scores.var() > scaled_scores.var(), \
            "Scaling should reduce variance of attention scores"

    def test_different_q_k_v_values(self):
        """Test with different Q, K, V tensors."""
        Q = torch.randn(2, 8, 32)
        K = torch.randn(2, 8, 32)
        V = torch.randn(2, 8, 32)

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == V.shape, "Output should have same shape as V"
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8))


class TestCreatePaddingMask:
    """Test cases for padding mask creation."""

    def test_basic_mask(self):
        """Test basic padding mask creation."""
        seq = torch.tensor([
            [5, 234, 67, 891, 0, 0, 0],    # 4 real tokens, 3 padding
            [123, 456, 0, 0, 0, 0, 0],     # 2 real tokens, 5 padding
            [8, 345, 22, 678, 99, 0, 0]    # 5 real tokens, 2 padding
        ])

        mask = create_padding_mask(seq, pad_token_id=0)

        expected = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0]
        ])

        assert torch.equal(mask, expected), "Mask should correctly identify padding"

    def test_no_padding(self):
        """Test when there's no padding."""
        seq = torch.tensor([[1, 2, 3, 4, 5]])
        mask = create_padding_mask(seq)

        assert torch.all(mask == 1), "Mask should be all 1s when no padding"

    def test_all_padding(self):
        """Test when everything is padding."""
        seq = torch.zeros(2, 10, dtype=torch.long)
        mask = create_padding_mask(seq)

        assert torch.all(mask == 0), "Mask should be all 0s when all padding"

    def test_custom_pad_id(self):
        """Test with custom padding token ID."""
        seq = torch.tensor([[1, 2, 3, 99, 99], [5, 99, 99, 99, 99]])
        mask = create_padding_mask(seq, pad_token_id=99)

        expected = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])

        assert torch.equal(mask, expected)


class TestMultiHeadAttention:
    """Test cases for multi-head attention."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 10

        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attention_weights = mha(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model), \
            f"Expected output shape ({batch_size}, {seq_len}, {d_model}), got {output.shape}"

        # Check attention weights shape
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"Expected attention shape ({batch_size}, {num_heads}, {seq_len}, {seq_len})"

    def test_with_mask(self):
        """Test multi-head attention with mask."""
        d_model = 64
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(2, 10, d_model)
        mask = torch.ones(2, 10)
        mask[0, 6:] = 0
        mask[1, 4:] = 0

        output, attention_weights = mha(x, mask)

        assert output.shape == (2, 10, d_model)

        # Check that masked positions get near-zero attention
        # attention_weights is [batch, num_heads, seq_len, seq_len]
        for batch_idx in range(2):
            mask_start = 6 if batch_idx == 0 else 4
            # Check that attention to masked positions is ~0
            assert attention_weights[batch_idx, :, :, mask_start:].max() < 1e-6

    def test_split_combine_heads(self):
        """Test split_heads and combine_heads are inverse operations."""
        d_model = 256
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(3, 15, d_model)

        # Split then combine should give back original
        split = mha.split_heads(x)
        combined = mha.combine_heads(split)

        assert torch.allclose(x, combined), \
            "Splitting then combining should recover original tensor"

    def test_different_num_heads(self):
        """Test with different numbers of heads."""
        d_model = 128
        batch_size = 4
        seq_len = 20

        for num_heads in [1, 2, 4, 8, 16]:
            mha = MultiHeadAttention(d_model, num_heads)
            x = torch.randn(batch_size, seq_len, d_model)

            output, _ = mha(x)

            assert output.shape == (batch_size, seq_len, d_model)

    def test_invalid_num_heads(self):
        """Test that invalid num_heads raises error."""
        with pytest.raises(AssertionError):
            # d_model=128 is not divisible by num_heads=7
            MultiHeadAttention(d_model=128, num_heads=7)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        d_model = 64
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output, _ = mha(x)

        # Backpropagate
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Gradients should flow to input"
        assert mha.W_q.weight.grad is not None, "Gradients should flow to W_q"
        assert mha.W_k.weight.grad is not None, "Gradients should flow to W_k"
        assert mha.W_v.weight.grad is not None, "Gradients should flow to W_v"
        assert mha.W_o.weight.grad is not None, "Gradients should flow to W_o"

    def test_single_head_vs_multihead(self):
        """Compare single head vs multi-head (architectural test)."""
        d_model = 64
        x = torch.randn(2, 8, d_model)

        # Single head
        mha_single = MultiHeadAttention(d_model, num_heads=1)
        output_single, _ = mha_single(x)

        # Multiple heads
        mha_multi = MultiHeadAttention(d_model, num_heads=4)
        output_multi, _ = mha_multi(x)

        # Both should produce valid outputs of the same shape
        assert output_single.shape == output_multi.shape == (2, 8, d_model)

    def test_large_dimensions(self):
        """Test with BERT-like dimensions."""
        d_model = 768
        num_heads = 12
        batch_size = 8
        seq_len = 128

        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attention_weights = mha(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
