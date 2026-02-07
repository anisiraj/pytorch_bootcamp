"""Examples of different ways to import from nlp_utils package."""

# Method 1: Import from main package namespace
from nlp_utils import MaskedMeanPool

# Method 2: Import from specific module
from nlp_utils.models.pooling import MaskedMeanPool

# Method 3: Import the models subpackage
from nlp_utils import models

# Method 4: Import multiple items at once
from nlp_utils import (
    MaskedMeanPool,
    SentimentModel,
    Vocabulary,
    SimpleTokenizer,
    ReviewDataSet,
    collate_fn,
    Trainer,
)

# Method 5: Import with alias
from nlp_utils import MaskedMeanPool as MeanPool

# Method 6: Import entire module
import nlp_utils


def demonstrate_imports():
    """Show that all import methods work."""
    import torch

    # Using direct import
    pool1 = MaskedMeanPool()

    # Using module import
    pool2 = models.MaskedMeanPool()

    # Using package import
    pool3 = nlp_utils.MaskedMeanPool()

    # Using aliased import
    pool4 = MeanPool()

    # They all work the same way
    embeddings = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5)

    result1 = pool1(embeddings, mask)
    result2 = pool2(embeddings, mask)
    result3 = pool3(embeddings, mask)
    result4 = pool4(embeddings, mask)

    print("All import methods work correctly!")
    print(f"Result shape: {result1.shape}")

    # Verify they all produce the same result
    assert torch.allclose(result1, result2)
    assert torch.allclose(result2, result3)
    assert torch.allclose(result3, result4)
    print("All methods produce identical results!")


if __name__ == "__main__":
    demonstrate_imports()

    # Show what's available in nlp_utils
    print("\nAvailable in nlp_utils package:")
    print(nlp_utils.__all__)
