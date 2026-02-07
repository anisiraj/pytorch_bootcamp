# Day 2: Attention from Scratch - Quick Reference

## Components Now in nlp_utils

The attention mechanisms from your notebook have been extracted into the `nlp_utils` package for reuse!

### Import Everything You Need

```python
from nlp_utils import (
    # Attention components
    scaled_dot_product_attention,
    create_padding_mask,
    MultiHeadAttention,

    # Other utilities (already available)
    MaskedMeanPool,
    Vocabulary,
    SimpleTokenizer,
    # ... etc
)
```

## Quick Usage Examples

### 1. Scaled Dot-Product Attention

```python
import torch
from nlp_utils import scaled_dot_product_attention

# Create Q, K, V
Q = K = V = torch.randn(2, 10, 64)  # [batch, seq_len, d_k]

# Apply attention
output, weights = scaled_dot_product_attention(Q, K, V)

# With padding mask
mask = torch.ones(2, 10)
mask[0, 7:] = 0  # Mask last 3 positions
output, weights = scaled_dot_product_attention(Q, K, V, mask)
```

### 2. Create Padding Mask

```python
from nlp_utils import create_padding_mask

# Your sequence with padding (0 = padding token)
seq = torch.tensor([
    [5, 234, 67, 891, 0, 0, 0],    # 4 real, 3 padding
    [123, 456, 0, 0, 0, 0, 0],     # 2 real, 5 padding
])

# Create mask (1 = real, 0 = padding)
mask = create_padding_mask(seq, pad_token_id=0)
# Result: tensor([[1,1,1,1,0,0,0], [1,1,0,0,0,0,0]])
```

### 3. Multi-Head Attention

```python
from nlp_utils import MultiHeadAttention

# Create layer
d_model = 128
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

# Apply to input
x = torch.randn(32, 50, 128)  # [batch, seq_len, d_model]
output, attention_weights = mha(x)

# With mask
mask = torch.ones(32, 50)
output, attention_weights = mha(x, mask)
```

## Tests Available

Run tests from the project root:

```bash
# Test attention components
pytest tests/test_attention.py -v

# Test specific component
pytest tests/test_attention.py::TestMultiHeadAttention -v

# Test all
pytest tests/ -v
```

## Examples and Demos

Check out these files in the `examples/` directory:

- `examples/attention_demo.py` - Comprehensive attention demonstrations
- `examples/pooling_demo.py` - Pooling layer usage
- `examples/import_examples.py` - Different import patterns

Run them:
```bash
python examples/attention_demo.py
```

## Documentation

See detailed documentation in:
- `ATTENTION_SETUP.md` - Full attention setup and usage guide
- `POOLING_SETUP.md` - Pooling layers guide
- `tests/README.md` - Testing guide

## What's Tested

✅ 18 attention tests covering:
- Scaled dot-product attention (basic, masked, edge cases)
- Padding mask creation
- Multi-head attention (1-16 heads, masking, gradients)
- BERT-scale dimensions (768d, 12 heads)

✅ 9 pooling tests covering:
- Masked mean pooling
- Gradient flow
- Various batch sizes and dimensions

**All 27 tests pass!**

## Building Transformer Components

Now you can build the rest of the transformer:

```python
import torch.nn as nn
from nlp_utils import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        x = self.norm2(x + self.ffn(x))
        return x
```

## Continue Building

Next components to implement for full transformer:
1. ✅ Scaled Dot-Product Attention (done!)
2. ✅ Multi-Head Attention (done!)
3. ⬜ Positional Encoding
4. ⬜ Feed-Forward Network
5. ⬜ Transformer Encoder Block
6. ⬜ Full Transformer Classifier

Happy coding! 🚀
