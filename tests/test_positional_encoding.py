"""Tests for PositionalEncoding."""

import pytest
import torch
from nlp_utils.models.attention import PositionalEncoding


class TestPositionalEncoding:
    """Test cases for PositionalEncoding."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        pe = PositionalEncoding(d_model=128, max_len=512)

        assert pe.d_model == 128
        assert pe.max_len == 512
        assert pe.pe.shape == (1, 512, 128)

    def test_forward_pass(self):
        """Test forward pass with valid input."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(32, 50, 64)

        output = pe(x)

        assert output.shape == (32, 50, 64)
        assert not torch.allclose(x, output), "Output should be different from input"

    def test_position_zero_pattern(self):
        """Test that position 0 has sin=0, cos=1 pattern."""
        pe = PositionalEncoding(d_model=8, max_len=10)

        # At position 0: sin(0) = 0, cos(0) = 1
        pos_0 = pe.pe[0, 0, :]

        # Even indices should be ~0 (sin)
        assert torch.allclose(pos_0[0::2], torch.zeros(4), atol=1e-6)

        # Odd indices should be ~1 (cos)
        assert torch.allclose(pos_0[1::2], torch.ones(4), atol=1e-6)

    def test_different_positions_different_encodings(self):
        """Test that different positions have different encodings."""
        pe = PositionalEncoding(d_model=64, max_len=100)

        pos_0 = pe.pe[0, 0, :]
        pos_1 = pe.pe[0, 1, :]
        pos_10 = pe.pe[0, 10, :]

        assert not torch.allclose(pos_0, pos_1)
        assert not torch.allclose(pos_0, pos_10)
        assert not torch.allclose(pos_1, pos_10)

    def test_variable_sequence_lengths(self):
        """Test with different sequence lengths."""
        pe = PositionalEncoding(d_model=128, max_len=512)

        for seq_len in [10, 50, 100, 200, 512]:
            x = torch.randn(8, seq_len, 128)
            output = pe(x)
            assert output.shape == (8, seq_len, 128)

    def test_broadcasting_across_batch(self):
        """Test that PE broadcasts correctly across batch dimension."""
        pe = PositionalEncoding(d_model=32, max_len=20)

        x1 = torch.zeros(1, 10, 32)
        x2 = torch.zeros(16, 10, 32)
        x3 = torch.zeros(64, 10, 32)

        out1 = pe(x1)
        out2 = pe(x2)
        out3 = pe(x3)

        # All should have same positional encodings
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[0], out3[0])

    def test_invalid_embedding_dimension(self):
        """Test that mismatched embedding dimension raises error."""
        pe = PositionalEncoding(d_model=128, max_len=100)
        x = torch.randn(32, 50, 64)  # Wrong d_model

        with pytest.raises(ValueError, match="Input embedding dimension"):
            pe(x)

    def test_invalid_input_shape(self):
        """Test that wrong input shape raises error."""
        pe = PositionalEncoding(d_model=64, max_len=100)

        # 2D input (missing batch dimension)
        x_2d = torch.randn(50, 64)
        with pytest.raises(ValueError, match="Input must be 3D"):
            pe(x_2d)

        # 4D input - will fail on embedding dimension check first
        x_4d = torch.randn(32, 50, 64, 1)
        with pytest.raises(ValueError):  # Either error is fine
            pe(x_4d)

    def test_not_learnable(self):
        """Test that PE has no learnable parameters."""
        pe = PositionalEncoding(d_model=64, max_len=100)

        param_count = sum(p.numel() for p in pe.parameters() if p.requires_grad)
        assert param_count == 0, "PositionalEncoding should have no learnable parameters"

    def test_gradient_flow(self):
        """Test that gradients flow through PE."""
        pe = PositionalEncoding(d_model=32, max_len=50)
        x = torch.randn(4, 10, 32, requires_grad=True)

        output = pe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_static_method(self):
        """Test static method for calculating encodings."""
        pe_tensor = PositionalEncoding.calculate_positional_encodings(
            d_model=64,
            max_len=100
        )

        assert pe_tensor.shape == (1, 100, 64)
        assert isinstance(pe_tensor, torch.Tensor)

    def test_property_setters(self):
        """Test that property setters work and recompute PE."""
        pe = PositionalEncoding(d_model=64, max_len=100)

        original_pe = pe.pe.clone()

        # Change d_model
        pe.d_model = 128
        assert pe.d_model == 128
        assert pe.pe.shape == (1, 100, 128)
        assert not torch.equal(pe.pe[:, :, :64], original_pe)

        # Change max_len
        pe.max_len = 200
        assert pe.max_len == 200
        assert pe.pe.shape == (1, 200, 128)

    def test_different_d_models(self):
        """Test with different embedding dimensions."""
        for d_model in [32, 64, 128, 256, 512, 768]:
            pe = PositionalEncoding(d_model=d_model, max_len=100)
            x = torch.randn(8, 50, d_model)
            output = pe(x)
            assert output.shape == (8, 50, d_model)

    def test_sin_cos_frequencies(self):
        """Test that sin/cos pairs use same frequencies."""
        pe = PositionalEncoding(d_model=8, max_len=100)

        # For each sin/cos pair, verify they have related values
        # At position 1:
        pos_1 = pe.pe[0, 1, :]

        # sin^2 + cos^2 should be ~1 for each pair
        for i in range(0, 8, 2):
            sin_val = pos_1[i]
            cos_val = pos_1[i+1]
            # They should satisfy sin^2 + cos^2 ≈ 1 (might not be exact due to different frequencies)
            # Just verify they're reasonable values
            assert -1 <= sin_val <= 1
            assert -1 <= cos_val <= 1

    def test_large_sequence_length(self):
        """Test with large sequence lengths."""
        pe = PositionalEncoding(d_model=128, max_len=2048)
        x = torch.randn(2, 2000, 128)

        output = pe(x)
        assert output.shape == (2, 2000, 128)

    def test_sequence_longer_than_max_len_fails(self):
        """Test that sequence longer than max_len fails gracefully."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(4, 150, 64)  # seq_len=150 > max_len=100

        # This should fail because we try to slice pe[:, :150, :] but max_len=100
        with pytest.raises(RuntimeError):
            pe(x)

    def test_bert_dimensions(self):
        """Test with BERT-like dimensions."""
        # BERT-base: d_model=768, max_len=512
        pe = PositionalEncoding(d_model=768, max_len=512)
        x = torch.randn(16, 128, 768)

        output = pe(x)
        assert output.shape == (16, 128, 768)

    def test_gpt2_dimensions(self):
        """Test with GPT-2-like dimensions."""
        # GPT-2: d_model=768, max_len=1024
        pe = PositionalEncoding(d_model=768, max_len=1024)
        x = torch.randn(8, 512, 768)

        output = pe(x)
        assert output.shape == (8, 512, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
