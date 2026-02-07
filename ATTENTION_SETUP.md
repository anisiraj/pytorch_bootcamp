# Attention Mechanisms Integration Summary

The attention mechanisms from day2 transformer notebook have been successfully extracted and integrated into the `nlp_utils` package with comprehensive tests and examples.

## What Was Done

### 1. Package Structure
Attention components are organized in the nlp_utils package:
```
nlp_utils/
├── models/
│   ├── __init__.py          # Exports attention components
│   ├── attention.py         # Contains attention implementations
│   ├── pooling.py           # Pooling layers
│   └── sentiment.py         # Sentiment model
└── __init__.py              # Main package exports
```

### 2. Components Implemented

#### `scaled_dot_product_attention(Q, K, V, mask=None)`
- Computes attention scores: `softmax(QK^T / √d_k) V`
- Supports optional padding mask
- Returns output and attention weights

#### `create_padding_mask(seq, pad_token_id=0)`
- Creates binary mask from input sequences
- 1 for real tokens, 0 for padding
- Customizable padding token ID

#### `MultiHeadAttention(d_model, num_heads)`
- Multi-head attention mechanism
- Methods:
  - `split_heads()` - Split embeddings into multiple heads
  - `combine_heads()` - Merge heads back together
  - `forward()` - Apply multi-head attention
- Supports masking
- Returns output and attention weights

### 3. Tests Created
Comprehensive test suite in `tests/test_attention.py` with **18 test cases**:

**Scaled Dot-Product Attention Tests:**
- Basic attention without mask
- Attention with padding mask
- Edge cases (all masked, single token)
- Scaling effect verification
- Different Q, K, V values

**Padding Mask Tests:**
- Basic mask creation
- No padding scenario
- All padding scenario
- Custom padding token ID

**Multi-Head Attention Tests:**
- Basic forward pass
- Attention with mask
- Split/combine heads operations
- Different numbers of heads (1, 2, 4, 8, 16)
- Invalid configuration handling
- Gradient flow verification
- Large BERT-like dimensions (768d, 12 heads)

**All 18 tests pass!** ✅

### 4. Examples Created
Created `examples/attention_demo.py` with 6 demonstrations:
1. Scaled dot-product attention basics
2. Attention with padding mask
3. Multi-head attention usage
4. Multi-head attention with mask
5. Comparison of different head counts
6. Gradient flow verification

### 5. Exports Updated
Updated package exports in:
- `nlp_utils/models/__init__.py`
- `nlp_utils/__init__.py`

## How to Use

### Import Components
```python
from nlp_utils import (
    scaled_dot_product_attention,
    create_padding_mask,
    MultiHeadAttention,
)
```

### Scaled Dot-Product Attention
```python
import torch

# Create Q, K, V tensors
Q = K = V = torch.randn(2, 10, 64)  # [batch, seq_len, d_k]

# Apply attention
output, weights = scaled_dot_product_attention(Q, K, V)

# With mask
mask = torch.ones(2, 10)
mask[0, 7:] = 0  # Mask last 3 positions
output, weights = scaled_dot_product_attention(Q, K, V, mask)
```

### Padding Mask Creation
```python
# Create mask from sequence
seq = torch.tensor([[5, 234, 67, 0, 0]])  # 3 real tokens, 2 padding
mask = create_padding_mask(seq, pad_token_id=0)
# Result: tensor([[1, 1, 1, 0, 0]])
```

### Multi-Head Attention
```python
# Create multi-head attention layer
d_model = 128
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

# Apply to input
x = torch.randn(32, 50, 128)  # [batch, seq_len, d_model]
output, attention_weights = mha(x)

# With mask
mask = torch.ones(32, 50)
mask[:, 40:] = 0  # Mask positions 40-49
output, attention_weights = mha(x, mask)
```

## Running Tests

```bash
# Run all attention tests
pytest tests/test_attention.py -v

# Run specific test
pytest tests/test_attention.py::TestMultiHeadAttention::test_basic_forward -v

# Run all tests
pytest tests/ -v
```

## Running Examples

```bash
# Run attention demonstrations
python examples/attention_demo.py
```

## Key Features

### Scaled Dot-Product Attention
- Proper scaling by √d_k to prevent vanishing gradients
- Efficient masked fill with -inf for padding
- Returns both output and attention weights
- Works with batched inputs

### Multi-Head Attention
- Splits d_model into num_heads parallel attention layers
- Each head has dimension d_model / num_heads
- Learns different representation subspaces
- Combines heads via concatenation + projection
- Full gradient support for training

### Technical Details

**Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-Head Process:**
1. Project input to Q, K, V using learned matrices
2. Split into num_heads parallel heads
3. Apply scaled dot-product attention to each head
4. Concatenate all heads
5. Final projection with W_o

**Shapes:**
```python
# Scaled dot-product attention
Q, K, V: [batch, seq_len, d_k]
output:  [batch, seq_len, d_k]
weights: [batch, seq_len, seq_len]

# Multi-head attention
input:   [batch, seq_len, d_model]
output:  [batch, seq_len, d_model]
weights: [batch, num_heads, seq_len, seq_len]
```

## Integration in Models

Example usage in a transformer model:

```python
import torch.nn as nn
from nlp_utils import MultiHeadAttention, MaskedMeanPool

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
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
```

## Next Steps

Now you can:
1. Use attention mechanisms in your transformer models
2. Build complete transformer encoder/decoder blocks
3. Experiment with different numbers of heads
4. Visualize attention patterns using the returned weights
5. Extend with positional encoding and full transformer architecture

## Files Added/Modified

### Created
- `nlp_utils/models/attention.py` - Attention implementations
- `tests/test_attention.py` - 18 comprehensive tests
- `examples/attention_demo.py` - 6 demonstration examples
- `ATTENTION_SETUP.md` (this file)

### Modified
- `nlp_utils/models/__init__.py` - Added attention exports
- `nlp_utils/__init__.py` - Added attention exports

## Common Gotchas (Documented in Code)

The implementation avoids common pitfalls:
- ✅ Uses `.transpose(-2, -1)` instead of `.T` for batched tensors
- ✅ Uses `math.sqrt()` for int, not `torch.sqrt()`
- ✅ Properly expands masks for broadcasting
- ✅ Uses `torch.allclose()` for float comparisons
- ✅ Includes `.contiguous()` before `.view()` operations

## Test Coverage Summary

```
tests/test_attention.py::TestScaledDotProductAttention     6 tests ✅
tests/test_attention.py::TestCreatePaddingMask             4 tests ✅
tests/test_attention.py::TestMultiHeadAttention            8 tests ✅
────────────────────────────────────────────────────────────────────
Total                                                     18 tests ✅
```

All components are production-ready and thoroughly tested!
