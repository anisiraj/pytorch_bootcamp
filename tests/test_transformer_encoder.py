"""Tests for TransformerEncoderBlock."""

import pytest
import torch
from nlp_utils import TransformerEncoderBlock, MultiHeadAttention, FeedForward


@pytest.fixture
def d_model():
    """Standard model dimension for testing."""
    return 128


@pytest.fixture
def num_heads():
    """Standard number of attention heads."""
    return 8


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 16


@pytest.fixture
def seq_len():
    """Standard sequence length for testing."""
    return 50


@pytest.fixture
def mha(d_model, num_heads):
    """Create multi-head attention module."""
    return MultiHeadAttention(d_model=d_model, num_heads=num_heads)


@pytest.fixture
def ff(d_model):
    """Create feed-forward module."""
    return FeedForward(d_model=d_model, d_ff=512, dropout_rate=0.1)


@pytest.fixture
def encoder_block(mha, ff):
    """Create transformer encoder block."""
    return TransformerEncoderBlock(mha=mha, ff=ff)


def test_basic_forward_pass(encoder_block, batch_size, seq_len, d_model):
    """Test basic forward pass preserves shape."""
    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder_block(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_forward_with_mask(encoder_block, batch_size, seq_len, d_model):
    """Test forward pass with padding mask."""
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 40:] = 0  # Last 10 positions are padding

    output = encoder_block(x, mask=mask)

    assert output.shape == (batch_size, seq_len, d_model)


def test_d_model_property(encoder_block, d_model):
    """Test d_model property is accessible."""
    assert encoder_block.d_model == d_model


def test_none_mha_raises_error(ff):
    """Test that None mha raises AssertionError."""
    with pytest.raises(AssertionError, match="MultiHeadAttention.*cannot be None"):
        TransformerEncoderBlock(mha=None, ff=ff)


def test_none_ff_raises_error(mha):
    """Test that None ff raises AssertionError."""
    with pytest.raises(AssertionError, match="FeedForward.*cannot be None"):
        TransformerEncoderBlock(mha=mha, ff=None)


def test_mismatched_d_model_raises_error():
    """Test that mismatched d_model between mha and ff raises error."""
    mha = MultiHeadAttention(d_model=128, num_heads=8)
    ff = FeedForward(d_model=64, d_ff=256)  # Different d_model

    with pytest.raises(AssertionError, match="d_model mismatch"):
        TransformerEncoderBlock(mha=mha, ff=ff)


def test_wrong_input_dimension_raises_error(encoder_block, batch_size, seq_len):
    """Test that wrong input dimension raises error."""
    x_bad = torch.randn(batch_size, seq_len, 64)  # Wrong d_model

    with pytest.raises(AssertionError, match="Input embedding dim.*must match d_model"):
        encoder_block(x_bad)


def test_output_different_from_input(encoder_block, batch_size, seq_len, d_model):
    """Test that output is transformed (not identity)."""
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    encoder_block.eval()
    with torch.no_grad():
        output = encoder_block(x)

    assert not torch.allclose(output, x)


@pytest.mark.parametrize("seq_len", [10, 25, 50, 100])
def test_different_sequence_lengths(encoder_block, d_model, seq_len):
    """Test that encoder block works with different sequence lengths."""
    x = torch.randn(4, seq_len, d_model)
    output = encoder_block(x)

    assert output.shape == (4, seq_len, d_model)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_different_batch_sizes(encoder_block, d_model, batch_size):
    """Test that encoder block works with different batch sizes."""
    x = torch.randn(batch_size, 20, d_model)
    output = encoder_block(x)

    assert output.shape == (batch_size, 20, d_model)


def test_gradient_flow(encoder_block, batch_size, seq_len, d_model):
    """Test that gradients can flow through the encoder block."""
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output = encoder_block(x)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_deterministic_in_eval_mode(encoder_block, batch_size, seq_len, d_model):
    """Test that forward pass is deterministic in eval mode."""
    encoder_block.eval()

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output1 = encoder_block(x)
        output2 = encoder_block(x)

    assert torch.allclose(output1, output2)


def test_layer_norms_exist(encoder_block):
    """Test that layer normalization layers are properly initialized."""
    assert hasattr(encoder_block, 'ln1')
    assert hasattr(encoder_block, 'ln2')
    assert encoder_block.ln1 is not encoder_block.ln2