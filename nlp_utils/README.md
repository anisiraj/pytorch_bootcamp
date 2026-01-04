# NLP Utils

Reusable NLP utilities for PyTorch text classification tasks including data processing, models, and training.

## Installation

The package is installed in editable mode, so any changes to the code will be immediately reflected:

```bash
uv pip install -e .
```

## Quick Start

```python
import torch
import torch.nn as nn
import torchmetrics
from nlp_utils import (
    SimpleTokenizer, ReviewDataSet, collate_fn,
    extract_imdb_sample_as_dict, SentimentModel, Trainer
)
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. Load and prepare data
ds = load_dataset("stanfordnlp/imdb")
train, test = extract_imdb_sample_as_dict(ds, train_size=5000, test_size=1000)

# 2. Build tokenizer vocabulary
tokenizer = SimpleTokenizer(lowercase=True, remove_punctuation=True, min_freq=2)
tokenizer.build_vocab(train['text'])

# 3. Create datasets and loaders
train_dataset = ReviewDataSet(target=train, tokenizer=tokenizer)
test_dataset = ReviewDataSet(target=test, tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 4. Create model and trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentModel(tokenizer, embedding_dim=256, hidden_dim=128, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

metrics = {
    'acc': torchmetrics.Accuracy(task='multiclass', num_classes=2),
    'f1': torchmetrics.F1Score(task='multiclass', num_classes=2)
}

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    optimizer=optimizer,
    loss_fn=nn.CrossEntropyLoss(),
    device=device,
    metrics=metrics
)

# 5. Train!
history = trainer.fit(num_epochs=5)
```

## Package Structure

```
nlp_utils/
├── __init__.py              # Main exports
├── tokenization/
│   ├── protocol.py          # TokenizerProtocol interface
│   ├── simple_tokenizer.py  # SimpleTokenizer (tokenization + vocabulary)
│   ├── tokenizer.py         # Legacy tokenize function
│   └── vocabulary.py        # Legacy Vocabulary class
├── data/
│   ├── dataset.py           # ReviewDataSet and collate_fn
│   └── utils.py             # Helper functions for data processing
├── models/
│   ├── pooling.py           # MaskedMeanPool layer
│   └── sentiment.py         # SentimentModel for text classification
└── training/
    └── trainer.py           # Trainer class with torchmetrics support
```

## Components

### SimpleTokenizer

Combined tokenization and vocabulary management following HuggingFace patterns. Handles tokenization, vocabulary building, encoding, and decoding in a single class.

```python
from nlp_utils import SimpleTokenizer

# Create tokenizer with configuration
tokenizer = SimpleTokenizer(
    lowercase=True,
    remove_punctuation=True,
    min_freq=2,
    min_length=1
)

# Build vocabulary from texts
tokenizer.build_vocab(['hello world', 'hello there', 'world'])

# Tokenize text
tokens = tokenizer.tokenize("Hello, World!")
# ['hello', 'world']

# Encode text to indices
indices = tokenizer.encode('hello world')

# Decode indices back to tokens
tokens = tokenizer.decode(indices)

# Access vocabulary size
vocab_size = tokenizer.vocab_size

# Access special token IDs
pad_id = tokenizer.pad_token_id  # 0
unk_id = tokenizer.unk_token_id  # 1

# Save/load tokenizer state
tokenizer.save('tokenizer.pkl')
loaded_tokenizer = SimpleTokenizer.load('tokenizer.pkl')
```

### TokenizerProtocol

Protocol interface that enables compatibility with any tokenizer (including HuggingFace tokenizers). Any tokenizer implementing this protocol can be used with the models and datasets.

```python
from nlp_utils import TokenizerProtocol, SimpleTokenizer
from transformers import AutoTokenizer

# SimpleTokenizer implements the protocol
tokenizer = SimpleTokenizer()
assert isinstance(tokenizer, TokenizerProtocol)  # True

# HuggingFace tokenizers also work (they have vocab_size, pad_token_id, encode, decode)
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Can be used with SentimentModel and ReviewDataSet

# Required attributes and methods:
# - vocab_size: int
# - pad_token_id: int
# - encode(text: str) -> list[int]
# - decode(ids: list[int]) -> list[str]
```

### ReviewDataSet

PyTorch Dataset for text classification tasks. Accepts any tokenizer implementing TokenizerProtocol.

```python
from nlp_utils import ReviewDataSet, SimpleTokenizer

# With SimpleTokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)

dataset = ReviewDataSet(
    target={'text': texts, 'label': labels},
    tokenizer=tokenizer,
    max_length=512
)

# Also works with HuggingFace tokenizers
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = ReviewDataSet(target=data, tokenizer=hf_tokenizer)
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

### MaskedMeanPool

Pooling layer that averages token embeddings while ignoring padding.

```python
from nlp_utils import MaskedMeanPool
import torch

pool = MaskedMeanPool()
embeddings = torch.randn(32, 128, 256)  # [batch, seq_len, emb_dim]
mask = torch.ones(32, 128)               # [batch, seq_len]
pooled = pool(embeddings, mask)          # [batch, emb_dim]
```

### SentimentModel

Two-layer sentiment classification model with masked mean pooling and dropout regularization. Accepts any tokenizer implementing TokenizerProtocol.

```python
from nlp_utils import SentimentModel, SimpleTokenizer

# With SimpleTokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)

model = SentimentModel(
    tokenizer=tokenizer,
    embedding_dim=256,
    hidden_dim=128,
    num_categories=2,
    dropout=0.3  # Add dropout for regularization
)

batch = {
    'input_ids': torch.randint(0, 1000, (32, 128)),
    'mask': torch.ones(32, 128)
}
logits = model(batch)  # [32, 2]
```

### Trainer

Complete training loop with validation, metrics, and history tracking.

```python
from nlp_utils import Trainer
import torch.nn as nn
import torchmetrics

# Define metrics
metrics = {
    'acc': torchmetrics.Accuracy(task='multiclass', num_classes=2),
    'f1': torchmetrics.F1Score(task='multiclass', num_classes=2),
    'precision': torchmetrics.Precision(task='multiclass', num_classes=2)
}

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    loss_fn=nn.CrossEntropyLoss(),
    device=device,
    metrics=metrics,
    grad_clip_norm=1.0
)

# Train
history = trainer.fit(num_epochs=5)

# Access history
print(history['train_loss'])  # List of training losses
print(history['val_acc'])     # List of validation accuracies
```

## Development

Since the package is installed in editable mode, you can modify the source files and the changes will be immediately available in your notebooks without reinstalling.