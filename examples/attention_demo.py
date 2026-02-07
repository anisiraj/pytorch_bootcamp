"""Demonstration of using attention mechanisms from nlp_utils."""

import torch
from nlp_utils import (
    scaled_dot_product_attention,
    create_padding_mask,
    MultiHeadAttention,
)


def demo_scaled_attention():
    """Demonstrate scaled dot-product attention."""
    print("=" * 60)
    print("Scaled Dot-Product Attention Demo")
    print("=" * 60)

    # Create sample Q, K, V tensors
    batch_size = 2
    seq_len = 8
    d_k = 64

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    print(f"\nInput shapes:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")

    # Apply attention without mask
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights sum to 1: {torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len))}")

    # Show first attention pattern
    print(f"\nFirst sequence attention pattern (position 0):")
    print(attention_weights[0, 0, :].round(decimals=3))


def demo_attention_with_mask():
    """Demonstrate attention with padding mask."""
    print("\n" + "=" * 60)
    print("Attention with Padding Mask Demo")
    print("=" * 60)

    # Create sequences with padding
    seq_tensor = torch.tensor([
        [5, 234, 67, 891, 12, 45, 0, 0],    # 6 real tokens, 2 padding
        [123, 456, 789, 0, 0, 0, 0, 0],     # 3 real tokens, 5 padding
    ])

    # Create padding mask
    mask = create_padding_mask(seq_tensor, pad_token_id=0)

    print(f"\nSequence tensor:\n{seq_tensor}")
    print(f"\nPadding mask (1=real, 0=padding):\n{mask}")

    # Create embeddings (in practice, these come from an embedding layer)
    batch_size, seq_len = seq_tensor.shape
    d_k = 32
    Q = K = V = torch.randn(batch_size, seq_len, d_k)

    # Apply attention with mask
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"\nAttention weights (first sequence):")
    print(attention_weights[0].round(decimals=4))

    # Verify padding positions get zero attention
    print(f"\nMax attention to padding positions (seq 1): {attention_weights[0, :6, 6:].max():.2e}")
    print(f"Max attention to padding positions (seq 2): {attention_weights[1, :3, 3:].max():.2e}")


def demo_multi_head_attention():
    """Demonstrate multi-head attention."""
    print("\n" + "=" * 60)
    print("Multi-Head Attention Demo")
    print("=" * 60)

    d_model = 128
    num_heads = 8
    batch_size = 4
    seq_len = 16

    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model, num_heads)

    print(f"\nMulti-Head Attention Configuration:")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"head_dim: {d_model // num_heads}")

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")

    # Apply multi-head attention
    output, attention_weights = mha(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  [batch, num_heads, seq_len, seq_len]")

    # Show attention pattern for one head
    print(f"\nAttention pattern for head 0, sequence 0:")
    print(attention_weights[0, 0, :5, :5].round(decimals=3))  # Show 5x5 subset


def demo_multi_head_with_mask():
    """Demonstrate multi-head attention with masking."""
    print("\n" + "=" * 60)
    print("Multi-Head Attention with Mask Demo")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    mha = MultiHeadAttention(d_model, num_heads)

    # Create sequences with different lengths
    batch_size = 3
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)

    # Create mask: different lengths per sequence
    mask = torch.zeros(batch_size, seq_len)
    mask[0, :8] = 1   # 8 real tokens
    mask[1, :5] = 1   # 5 real tokens
    mask[2, :10] = 1  # 10 real tokens (no padding)

    print(f"\nMask (shows which positions are real):")
    print(mask)

    # Apply attention
    output, attention_weights = mha(x, mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Verify padding gets no attention
    for i in range(batch_size):
        num_real = int(mask[i].sum())
        if num_real < seq_len:  # Only check if there's actually padding
            max_attn_to_padding = attention_weights[i, :, :num_real, num_real:].max()
            print(f"Seq {i}: {num_real} real tokens, max attention to padding: {max_attn_to_padding:.2e}")
        else:
            print(f"Seq {i}: {num_real} real tokens, no padding")


def demo_comparison_heads():
    """Compare different numbers of attention heads."""
    print("\n" + "=" * 60)
    print("Comparing Different Numbers of Heads")
    print("=" * 60)

    d_model = 128
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, d_model)

    for num_heads in [1, 2, 4, 8, 16]:
        mha = MultiHeadAttention(d_model, num_heads)
        output, attention_weights = mha(x)

        print(f"\nnum_heads={num_heads:2d}: "
              f"output {output.shape}, "
              f"attention {attention_weights.shape}, "
              f"head_dim={d_model // num_heads}")


def demo_gradient_flow():
    """Demonstrate gradient flow through attention."""
    print("\n" + "=" * 60)
    print("Gradient Flow Demo")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    mha = MultiHeadAttention(d_model, num_heads)

    # Create input with gradient tracking
    x = torch.randn(2, 8, d_model, requires_grad=True)

    print(f"\nInput requires_grad: {x.requires_grad}")

    # Forward pass
    output, _ = mha(x)

    # Dummy loss and backward
    loss = output.sum()
    loss.backward()

    print(f"Gradients computed: {x.grad is not None}")
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"W_q gradient exists: {mha.W_q.weight.grad is not None}")
    print(f"W_k gradient exists: {mha.W_k.weight.grad is not None}")
    print(f"W_v gradient exists: {mha.W_v.weight.grad is not None}")
    print(f"W_o gradient exists: {mha.W_o.weight.grad is not None}")

    print("\n✓ Gradients flow correctly through all components!")


if __name__ == "__main__":
    demo_scaled_attention()
    demo_attention_with_mask()
    demo_multi_head_attention()
    demo_multi_head_with_mask()
    demo_comparison_heads()
    demo_gradient_flow()

    print("\n" + "=" * 60)
    print("All attention demos completed successfully!")
    print("=" * 60)
