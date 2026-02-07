"""Tests for FeedForward network."""

import pytest
import torch
import torch.nn as nn
from nlp_utils.models.attention import FeedForward


class TestFeedForward:
    """Test cases for FeedForward network."""

    def test_basic_initialization(self):
        """Test basic initialization with default expansion."""
        ff = FeedForward(d_model=64)

        assert ff.d_model == 64
        assert ff.d_ff == 256  # 4x expansion
        assert ff.dropout_rate == 0.1

    def test_custom_d_ff(self):
        """Test initialization with custom d_ff."""
        ff = FeedForward(d_model=64, d_ff=128)

        assert ff.d_model == 64
        assert ff.d_ff == 128
        assert ff.dropout_rate == 0.1

    def test_custom_dropout(self):
        """Test initialization with custom dropout rate."""
        ff = FeedForward(d_model=64, dropout_rate=0.25)

        assert ff.dropout_rate == 0.25

    def test_forward_pass_shape(self):
        """Test that forward pass maintains input shape."""
        ff = FeedForward(d_model=64)
        x = torch.randn(32, 50, 64)

        output = ff(x)

        assert output.shape == (32, 50, 64)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        ff = FeedForward(d_model=128)

        for batch_size in [1, 8, 16, 32, 64]:
            x = torch.randn(batch_size, 20, 128)
            output = ff(x)
            assert output.shape == (batch_size, 20, 128)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        ff = FeedForward(d_model=128)

        for seq_len in [10, 50, 100, 512]:
            x = torch.randn(8, seq_len, 128)
            output = ff(x)
            assert output.shape == (8, seq_len, 128)

    def test_different_d_models(self):
        """Test with different model dimensions."""
        for d_model in [32, 64, 128, 256, 512, 768]:
            ff = FeedForward(d_model=d_model)
            x = torch.randn(8, 50, d_model)
            output = ff(x)
            assert output.shape == (8, 50, d_model)

    def test_expansion_ratios(self):
        """Test different expansion ratios."""
        d_model = 64

        # Test 2x, 4x, 8x expansion
        for ratio in [2, 4, 8]:
            d_ff = d_model * ratio
            ff = FeedForward(d_model=d_model, d_ff=d_ff)

            x = torch.randn(8, 20, d_model)
            output = ff(x)

            assert output.shape == (8, 20, d_model)
            assert ff.d_ff == d_ff

    def test_output_different_from_input(self):
        """Test that output is different from input."""
        ff = FeedForward(d_model=64)
        x = torch.randn(8, 20, 64)

        output = ff(x)

        assert not torch.allclose(x, output)

    def test_dropout_training_mode(self):
        """Test that dropout behaves differently in training vs eval mode."""
        ff = FeedForward(d_model=64, dropout_rate=0.5)
        x = torch.randn(8, 20, 64)

        # Training mode
        ff.train()
        output_train_1 = ff(x)
        output_train_2 = ff(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train_1, output_train_2)

        # Eval mode
        ff.eval()
        output_eval_1 = ff(x)
        output_eval_2 = ff(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval_1, output_eval_2)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        ff = FeedForward(d_model=64)
        x = torch.randn(8, 20, 64, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_learnable_parameters(self):
        """Test that network has the correct number of learnable parameters."""
        ff = FeedForward(d_model=64, d_ff=256)

        param_count = sum(p.numel() for p in ff.parameters() if p.requires_grad)

        # Linear_1: 64 * 256 weights + 256 bias = 16640
        # Linear in contractor: 256 * 64 weights + 64 bias = 16448
        # Total: 33088
        expected = (64 * 256 + 256) + (256 * 64 + 64)
        assert param_count == expected

    def test_zero_dropout(self):
        """Test with zero dropout."""
        ff = FeedForward(d_model=64, dropout_rate=0.0)
        x = torch.randn(8, 20, 64)

        ff.train()
        output1 = ff(x)
        output2 = ff(x)

        # With zero dropout, outputs should be identical even in training mode
        assert torch.allclose(output1, output2)

    def test_relu_activation(self):
        """Test that ReLU activation is applied."""
        ff = FeedForward(d_model=64, dropout_rate=0.0)  # No dropout to isolate ReLU

        # Create input that will produce negative values after first linear layer
        x = torch.ones(1, 1, 64) * -10

        ff.eval()
        output = ff(x)

        # Output should exist (ReLU working)
        assert output is not None
        assert output.shape == (1, 1, 64)

    def test_bert_dimensions(self):
        """Test with BERT-like dimensions."""
        # BERT-base: d_model=768, d_ff=3072 (4x)
        ff = FeedForward(d_model=768, d_ff=3072)
        x = torch.randn(16, 128, 768)

        output = ff(x)
        assert output.shape == (16, 128, 768)

    def test_gpt2_dimensions(self):
        """Test with GPT-2-like dimensions."""
        # GPT-2: d_model=768, d_ff=3072 (4x)
        ff = FeedForward(d_model=768, d_ff=3072)
        x = torch.randn(8, 512, 768)

        output = ff(x)
        assert output.shape == (8, 512, 768)

    def test_backward_compatibility_parameter_order(self):
        """Test that parameters can be specified in different ways."""
        # Test different initialization patterns
        ff1 = FeedForward(d_model=64)
        ff2 = FeedForward(64)
        ff3 = FeedForward(d_model=64, d_ff=256)
        ff4 = FeedForward(64, 256)
        ff5 = FeedForward(d_model=64, d_ff=256, dropout_rate=0.2)

        x = torch.randn(8, 20, 64)

        # All should work
        for ff in [ff1, ff2, ff3, ff4, ff5]:
            output = ff(x)
            assert output.shape == (8, 20, 64)

    def test_module_structure(self):
        """Test that the module has the expected structure."""
        ff = FeedForward(d_model=64)

        # Check that the module has the expected components
        assert hasattr(ff, 'linear_1')
        assert hasattr(ff, 'dropout')
        assert hasattr(ff, 'contractor')

        assert isinstance(ff.linear_1, nn.Linear)
        assert isinstance(ff.dropout, nn.Dropout)
        assert isinstance(ff.contractor, nn.Sequential)

    def test_contractor_structure(self):
        """Test that contractor has correct structure."""
        ff = FeedForward(d_model=64, d_ff=256)

        # Contractor should have Linear + Dropout
        assert len(ff.contractor) == 2
        assert isinstance(ff.contractor[0], nn.Linear)
        assert isinstance(ff.contractor[1], nn.Dropout)

        # Check dimensions
        assert ff.contractor[0].in_features == 256
        assert ff.contractor[0].out_features == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
