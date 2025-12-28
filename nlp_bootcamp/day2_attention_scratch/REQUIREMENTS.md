# Day 2: Transformers Are Not Magic

**Time**: 4 hours | **File**: `attention.py`

---

## Preparation (20 min)

**Derive on paper before coding**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Q: (batch, seq_len, d_k)
K: (batch, seq_len, d_k)
V: (batch, seq_len, d_v)
Output: (batch, seq_len, d_v)
```

Draw the dimensions. Understand each operation.

### Confidence Check
✅ I can derive attention math on paper.

---

## Task 1: Scaled Dot-Product Attention (60 min)

### What to Build

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Returns: (output, attention_weights)
    """
    # 1. scores = Q @ K^T
    # 2. scores = scores / sqrt(d_k)
    # 3. if mask: scores.masked_fill(mask == 0, -1e9)
    # 4. attention = softmax(scores)
    # 5. output = attention @ V
    pass

def create_padding_mask(seq, pad_idx=0):
    """Create mask: 1 for real tokens, 0 for padding"""
    pass
```

**Test it**:
```python
Q = K = V = torch.randn(2, 10, 64)
output, weights = scaled_dot_product_attention(Q, K, V)
assert weights.sum(dim=-1).allclose(torch.ones(2, 10))  # Should sum to 1
```

**Experiment**: Run with and without scaling. Compare variance.

---

## Task 2: Multi-Head Attention (60 min)

### What to Build

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # W_q, W_k, W_v, W_o projections
        pass

    def split_heads(self, x):
        """(batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)"""
        pass

    def combine_heads(self, x):
        """Reverse of split_heads"""
        pass

    def forward(self, Q, K, V, mask=None):
        # 1. Project Q, K, V
        # 2. Split into heads
        # 3. Apply attention
        # 4. Combine heads
        # 5. Final projection
        pass
```

### Confidence Check
✅ I can explain why multiple heads help.

---

## Task 3: Transformer Encoder (60 min)

**Dataset**: AG News, 5,000 samples

### What to Build

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pass

class FeedForward(nn.Module):
    def forward(self, x):
        # Linear → ReLU → Dropout → Linear → Dropout
        pass

class TransformerEncoderBlock(nn.Module):
    def forward(self, x, mask=None):
        # 1. x = LayerNorm(x + MultiHeadAttention(x))
        # 2. x = LayerNorm(x + FeedForward(x))
        pass

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2):
        # Embedding + PositionalEncoding + Encoder blocks + Classifier
        pass

    def forward(self, x, mask=None):
        # 1. Embed + position encode
        # 2. Pass through encoder blocks
        # 3. Mean pool + classify
        pass
```

**Train on AG News (4-class classification)**

### Confidence Check
✅ I can remove positional encoding and explain what breaks.

---

## Debugging Time (45 min)

**Experiments**:
1. Remove positional encoding → train → observe
2. Try num_heads = [1, 2, 4, 8]
3. Visualize attention:
   ```python
   import seaborn as sns
   sns.heatmap(attention_weights[0].detach().cpu())
   ```

---

## Reflection (25 min)

**Save to**: `../reflections/day2_reflection.md`

Write:
- Hardest part: math or implementation?
- Shape errors encountered
- Understanding of transformers now

---

## Expected Results

- Train acc: 70-80%
- Val acc: 65-75%
- Time: ~1-2 min/epoch on GPU

**If not working**: Check positional encoding, residual connections, layer norm
