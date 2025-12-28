# Day 1: I Own the Training Loop

**Time**: 4 hours | **File**: `train_from_scratch.py`

---

## Task 1: Text Pipeline (45 min)

**Dataset**: IMDb sentiment, 5,000 samples only

### What to Build

```python
def tokenize(text):
    """Lowercase, remove punctuation, split on whitespace"""
    pass

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}

    def build_from_texts(self, texts, min_freq=2):
        """Build vocab from texts"""
        pass

    def encode(self, text):
        """Text → list of indices"""
        pass

class IMDBDataset(Dataset):
    def __getitem__(self, idx):
        # Return: {'input_ids': tensor, 'mask': tensor, 'label': tensor}
        pass

def collate_fn(batch):
    """Pad sequences to same length"""
    pass
```

### Confidence Check
✅ I can explain why incorrect padding or masking breaks training.

---

## Task 2: Baseline Model (60 min)

**Architecture**: Embedding → Mean Pool → Linear

### What to Build

```python
class SentimentModel(nn.Module):
    def forward(self, input_ids, mask):
        # 1. Embed
        # 2. Mask padding
        # 3. Mean pool
        # 4. Classify
        pass

def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch with gradient clipping"""
    # Don't forget: torch.nn.utils.clip_grad_norm_
    pass

def validate(model, dataloader, device):
    """Validation loop"""
    pass
```

### Confidence Check
✅ I can change embedding size or max sequence length without panic.

---

## Task 3: Loss Mastery (45 min)

### What to Build

```python
def manual_cross_entropy(logits, targets):
    """Implement CE loss from scratch"""
    # 1. Softmax
    # 2. Log
    # 3. Negative log likelihood
    pass

def create_nan_scenario():
    """Intentionally create NaNs, then fix them"""
    # Try: LR=1.0, no gradient clipping, log(0)
    pass
```

### Confidence Check
✅ I know exactly where NaNs come from.

---

## Debugging Time (45 min)

**Experiments**:
1. Train with different embedding dims: [64, 128, 256]
2. Break gradient clipping → observe exploding gradients
3. Use LR=1.0 → observe NaNs → fix them

---

## Reflection (25 min)

**Save to**: `../reflections/day1_reflection.md`

Write:
- What broke and how you fixed it
- What you learned about training loops
- What still confuses you

---

## Expected Results

- Train acc: 75-80%
- Val acc: 70-75%
- Time: ~30 sec/epoch on GPU

**If < 60% accuracy**: Debug tokenization, padding, masking
