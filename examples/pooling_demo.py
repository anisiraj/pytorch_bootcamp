"""Demonstration of using pooling layers from nlp_utils."""

import torch
from nlp_utils import MaskedMeanPool
from nlp_utils.models.pooling import MaskedMeanPool as DirectImport


def basic_example():
    """Basic usage of MaskedMeanPool."""
    print("=" * 60)
    print("Basic MaskedMeanPool Example")
    print("=" * 60)

    # Initialize the pooling layer
    pool = MaskedMeanPool()

    # Create sample embeddings
    # Shape: [batch_size, seq_len, embedding_dim]
    batch_size = 2
    seq_len = 5
    emb_dim = 4

    embeddings = torch.randn(batch_size, seq_len, emb_dim)
    print(f"\nInput embeddings shape: {embeddings.shape}")
    print(f"Embeddings:\n{embeddings}\n")

    # Create mask (1 for real tokens, 0 for padding)
    mask = torch.ones(batch_size, seq_len)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask}\n")

    # Apply pooling
    pooled = pool(embeddings, mask)
    print(f"Pooled output shape: {pooled.shape}")
    print(f"Pooled output:\n{pooled}\n")


def padding_example():
    """Example with padding tokens."""
    print("=" * 60)
    print("MaskedMeanPool with Padding Example")
    print("=" * 60)

    pool = MaskedMeanPool()

    # Create embeddings with clear values for demonstration
    embeddings = torch.tensor([
        # Sequence 1: 3 real tokens, 2 padding
        [[1.0, 0.0],
         [2.0, 0.0],
         [3.0, 0.0],
         [999.0, 999.0],  # Padding (should be ignored)
         [999.0, 999.0]], # Padding (should be ignored)

        # Sequence 2: 2 real tokens, 3 padding
        [[4.0, 8.0],
         [6.0, 12.0],
         [999.0, 999.0],  # Padding
         [999.0, 999.0],  # Padding
         [999.0, 999.0]]  # Padding
    ])

    # Mask indicating real vs padding tokens
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],  # First 3 are real
        [1.0, 1.0, 0.0, 0.0, 0.0]   # First 2 are real
    ])

    print(f"\nInput embeddings shape: {embeddings.shape}")
    print(f"Embeddings:\n{embeddings}\n")
    print(f"Mask:\n{mask}\n")

    pooled = pool(embeddings, mask)

    print(f"Pooled output shape: {pooled.shape}")
    print(f"Pooled output:\n{pooled}")

    # Manual verification
    print("\nManual verification:")
    print(f"Sequence 1 mean: {embeddings[0, :3, :].mean(dim=0)}")
    print(f"Sequence 2 mean: {embeddings[1, :2, :].mean(dim=0)}")


def realistic_nlp_example():
    """Realistic NLP scenario example."""
    print("\n" + "=" * 60)
    print("Realistic NLP Scenario")
    print("=" * 60)

    pool = MaskedMeanPool()

    # Simulate a batch of sentences with varying lengths
    # Typical BERT-like dimensions
    batch_size = 4
    max_seq_len = 16  # Max sequence length in batch
    emb_dim = 128     # Embedding dimension

    # Random embeddings (in practice, these come from an embedding layer)
    embeddings = torch.randn(batch_size, max_seq_len, emb_dim)

    # Actual sequence lengths: [16, 12, 8, 5]
    seq_lengths = [16, 12, 8, 5]

    # Create mask based on actual lengths
    mask = torch.zeros(batch_size, max_seq_len)
    for i, length in enumerate(seq_lengths):
        mask[i, :length] = 1

    print(f"\nBatch size: {batch_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Embedding dimension: {emb_dim}")
    print(f"Actual sequence lengths: {seq_lengths}")
    print(f"\nMask (showing which positions are real):")
    print(mask)

    # Pool the embeddings
    pooled = pool(embeddings, mask)

    print(f"\nPooled output shape: {pooled.shape}")
    print(f"First pooled sentence (first 8 dims): {pooled[0, :8]}")

    # Verify that padding doesn't affect the result
    print("\nVerification - comparing with manual mean:")
    for i in range(batch_size):
        manual_mean = embeddings[i, :seq_lengths[i], :].mean(dim=0)
        pooled_mean = pooled[i]
        is_close = torch.allclose(manual_mean, pooled_mean, atol=1e-6)
        print(f"Sequence {i+1} (length {seq_lengths[i]}): Match = {is_close}")


def gradient_example():
    """Example showing gradient flow through pooling."""
    print("\n" + "=" * 60)
    print("Gradient Flow Example")
    print("=" * 60)

    pool = MaskedMeanPool()

    # Create embeddings with gradient tracking
    embeddings = torch.randn(2, 4, 3, requires_grad=True)

    mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],  # 3 real tokens
        [1.0, 1.0, 0.0, 0.0]   # 2 real tokens
    ])

    print(f"\nInput requires_grad: {embeddings.requires_grad}")

    # Forward pass
    pooled = pool(embeddings, mask)

    # Compute a dummy loss and backpropagate
    loss = pooled.sum()
    loss.backward()

    print(f"Pooled output shape: {pooled.shape}")
    print(f"Gradients computed: {embeddings.grad is not None}")

    if embeddings.grad is not None:
        print("\nGradient shape:", embeddings.grad.shape)
        print("\nGradients at padding positions should be zero:")
        print(f"Sequence 1, position 3 (padding): {embeddings.grad[0, 3, :]}")
        print(f"Sequence 2, positions 2-3 (padding): {embeddings.grad[1, 2:, :]}")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    padding_example()
    realistic_nlp_example()
    gradient_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
