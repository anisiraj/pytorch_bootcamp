# NLP Utils

Reusable NLP utilities for PyTorch text classification tasks.

## Installation

The package is installed in editable mode, so any changes to the code will be immediately reflected:

```bash
uv pip install -e .
```

## Usage

### In Jupyter Notebooks

```python
from nlp_utils import (
    tokenize,
    Vocabulary,
    ReviewDataSet,
    collate_fn,
    extract_imdb_sample_as_dict
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# 1. Load dataset and extract samples
ds = load_dataset("stanfordnlp/imdb")
train, test = extract_imdb_sample_as_dict(ds, train_size=5000, test_size=1000)

# 2. Build vocabulary
vocab = Vocabulary(tokenizer=tokenize)
vocab.build_from_texts(train['text'])

# 3. Create datasets
train_dataset = ReviewDataSet(target=train, vocab=vocab)
test_dataset = ReviewDataSet(target=test, vocab=vocab)

# 4. Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# 5. Use in training loop
for batch in train_loader:
    input_ids = batch['input_ids']  # [batch_size, seq_len]
    mask = batch['mask']            # [batch_size, seq_len]
    labels = batch['label']         # [batch_size]
    # Your training code here...
```

## Package Structure

```
nlp_utils/
├── __init__.py              # Main exports
├── tokenization/
│   ├── __init__.py
│   ├── tokenizer.py         # Simple whitespace tokenizer
│   └── vocabulary.py        # Vocabulary class for encoding/decoding
└── data/
    ├── __init__.py
    ├── dataset.py           # ReviewDataSet and collate_fn
    └── utils.py             # Helper functions for data processing
```

## Components

### Tokenizer

Simple whitespace-based tokenizer with options for lowercasing and punctuation removal.

```python
from nlp_utils import tokenize

tokens = tokenize("Hello, World!", lowercase=True, remove_punctuation=True)
# ['hello', 'world']
```

### Vocabulary

Handles encoding text to indices and decoding indices back to text.

```python
from nlp_utils import Vocabulary, tokenize

vocab = Vocabulary(tokenizer=tokenize, min_freq=2)
vocab.build_from_texts(['hello world', 'hello there'])

# Encode text to indices
indices = vocab.encode('hello world')

# Decode indices back to tokens
tokens = vocab.decode(indices)

# Get vocabulary size
print(len(vocab))
```

### ReviewDataSet

PyTorch Dataset for text classification tasks.

```python
from nlp_utils import ReviewDataSet

dataset = ReviewDataSet(
    target={'text': texts, 'label': labels},
    vocab=vocab,
    max_length=512
)
```

### collate_fn

Dynamic padding function for creating batches.

```python
from nlp_utils import collate_fn
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### extract_imdb_sample_as_dict

Extract and convert sampled IMDb dataset splits to dictionaries for easy processing.

```python
from nlp_utils import extract_imdb_sample_as_dict
from datasets import load_dataset

# Load full IMDb dataset
ds = load_dataset("stanfordnlp/imdb")

# Extract samples as dicts
train, test = extract_imdb_sample_as_dict(
    ds,
    train_size=5000,
    test_size=1000,
    seed=42,
    shuffle=True
)

# Returns dicts with 'text' and 'label' keys
print(type(train))  # <class 'dict'>
print(train.keys())  # dict_keys(['text', 'label'])
```

## Development

Since the package is installed in editable mode, you can modify the source files and the changes will be immediately available in your notebooks without reinstalling.