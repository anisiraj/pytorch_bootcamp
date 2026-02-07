# PyTorch Cheatsheet for NLP

## 🔥 Tensor Basics

### Creating Tensors
```python
import torch
import torch.nn as nn

# From data
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Common initializations
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
random = torch.randn(2, 3)  # Normal distribution
rand = torch.rand(2, 3)     # Uniform [0, 1)
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Like another tensor
x_zeros = torch.zeros_like(x)
x_ones = torch.ones_like(x)
```

### Tensor Operations
```python
# Shape operations
x.shape or x.size()  # Get shape
x.view(2, -1)        # Reshape (shares memory)
x.reshape(2, -1)     # Reshape (may copy)
x.squeeze()          # Remove dims of size 1
x.unsqueeze(0)       # Add dim at position 0
x.permute(1, 0, 2)   # Reorder dimensions
x.transpose(0, 1)    # Swap two dimensions

# Math operations
x + y, x - y, x * y, x / y  # Element-wise
x @ y or torch.matmul(x, y)  # Matrix multiplication
x.sum(), x.mean(), x.std()
x.max(), x.min(), x.argmax(), x.argmin()
torch.cat([x, y], dim=0)  # Concatenate
torch.stack([x, y], dim=0)  # Stack (adds new dim)

# Indexing
x[0]           # First row
x[:, 0]        # First column
x[0, 1:3]      # First row, columns 1-2
x[x > 0.5]     # Boolean indexing
```

### Device Management
```python
# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors
x = x.to(device)
x = x.cuda()  # To GPU
x = x.cpu()   # To CPU

# Create on device
x = torch.randn(3, 4, device=device)
```

### Gradient Tracking
```python
x = torch.randn(3, 4, requires_grad=True)
y = x * 2
z = y.mean()
z.backward()  # Compute gradients
print(x.grad)  # Access gradients

# Disable gradient tracking (for inference)
with torch.no_grad():
    y = model(x)

# Or use
torch.set_grad_enabled(False)
```

---

## 🧠 Neural Network Basics

### Defining Models
```python
import torch.nn as nn
import torch.nn.functional as F

# Method 1: nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# Method 2: Custom Module (Recommended)
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MyModel(784, 128, 10)
```

### Common Layers

#### For NLP
```python
# Embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
x = embedding(input_ids)  # (batch, seq_len) -> (batch, seq_len, embed_dim)

# LSTM / GRU
lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
               batch_first=True, dropout=0.2, bidirectional=True)
output, (hidden, cell) = lstm(x)  # x: (batch, seq_len, input_size)

gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, hidden = gru(x)

# Linear (Fully Connected)
fc = nn.Linear(in_features, out_features)

# Dropout
dropout = nn.Dropout(p=0.5)

# LayerNorm
layer_norm = nn.LayerNorm(normalized_shape)
```

#### Activation Functions
```python
nn.ReLU()
nn.Tanh()
nn.Sigmoid()
nn.GELU()  # Used in transformers
nn.LeakyReLU(negative_slope=0.01)
nn.Softmax(dim=-1)
F.log_softmax(x, dim=-1)  # For NLLLoss
```

### Model Utilities
```python
# Model to device
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers
for param in model.classifier.parameters():
    param.requires_grad = True

# Model summary
print(model)

# Save/Load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# Save entire model
torch.save(model, 'entire_model.pth')
model = torch.load('entire_model.pth')

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 📊 Data Loading

### Dataset Class
```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }

# Create dataset
dataset = TextDataset(texts, labels)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # For GPU
)

# Iterate
for batch in dataloader:
    texts = batch['text']
    labels = batch['label']
```

### Collate Function (for padding)
```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    return {
        'text': texts_padded,
        'label': labels
    }

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## 🎯 Training Loop

### Basic Training Loop
```python
import torch.optim as optim

# Setup
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()  # Set to training mode
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # Move to device
        inputs = batch['text'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
```

### With Validation
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        inputs = batch['text'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

---

## 🔧 Optimizers & Loss Functions

### Optimizers
```python
from torch import optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

# Adam (most common)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay fix)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Different learning rates for different layers
optimizer = optim.Adam([
    {'params': model.base.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### Learning Rate Schedulers
```python
from torch.optim.lr_scheduler import *

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Linear warmup then decay
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)

# Usage in training loop
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    scheduler.step()  # For most schedulers
    # scheduler.step(val_loss)  # For ReduceLROnPlateau
```

### Loss Functions
```python
# Classification
criterion = nn.CrossEntropyLoss()  # For multi-class (combines LogSoftmax + NLLLoss)
criterion = nn.NLLLoss()  # Negative log likelihood (use with log_softmax)
criterion = nn.BCELoss()  # Binary cross-entropy (use with sigmoid)
criterion = nn.BCEWithLogitsLoss()  # BCE with built-in sigmoid (more stable)

# Regression
criterion = nn.MSELoss()  # Mean squared error
criterion = nn.L1Loss()   # Mean absolute error
criterion = nn.SmoothL1Loss()  # Huber loss

# With class weights (for imbalanced data)
weights = torch.tensor([1.0, 2.0, 3.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Ignore padding tokens
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding
```

---

## 📈 Common NLP Patterns

### Sequence Padding
```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Pad sequences to same length
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
# tensor([[1, 2, 3],
#         [4, 5, 0],
#         [6, 0, 0]])

# For RNNs: pack for efficiency
lengths = torch.tensor([3, 2, 1])
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
output, hidden = lstm(packed)
output, _ = pad_packed_sequence(output, batch_first=True)
```

### Attention Mechanisms (From Scratch)

#### Scaled Dot-Product Attention
```python
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q, K, V: [batch, seq_len, d_k] or [batch, num_heads, seq_len, d_k]
        mask: [batch, seq_len] - 1 for real tokens, 0 for padding
    Returns:
        output: [batch, seq_len, d_k]
        attention_weights: [batch, seq_len, seq_len]
    """
    # Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]

    # Scale by sqrt(d_k)
    d_k = Q.shape[-1]  # NOT Q.shape(-1) - shape is attribute, not method
    scores = scores / math.sqrt(d_k)  # Use math.sqrt, NOT torch.sqrt(d_k)

    # Apply mask (set padding positions to -inf before softmax)
    if mask is not None:
        # Expand mask: [batch, seq_len] → [batch, 1, seq_len]
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to get attention weights (sum to 1 along last dim)
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = attention_weights @ V

    return output, attention_weights
```

#### Key Points for Attention
```python
# ✓ CORRECT: Use .transpose(-2, -1) for batched tensors
scores = Q @ K.transpose(-2, -1)

# ✗ WRONG: .T swaps ALL dimensions, breaks with batches
scores = Q @ K.T  # Only works for 2D matrices!

# ✓ CORRECT: Get dimension from shape attribute
d_k = Q.shape[-1]  # or Q.size(-1)

# ✗ WRONG: shape is not a method
d_k = Q.shape(-1)  # TypeError!

# ✓ CORRECT: Use math.sqrt for Python int
scores = scores / math.sqrt(d_k)

# ✗ WRONG: torch.sqrt expects tensor, not int
scores = scores / torch.sqrt(d_k)  # TypeError!

# ✓ CORRECT: Check attention weights sum to 1
assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len))

# Note: Use torch.allclose for floating point comparisons, not ==
```

#### Masking for Padding
```python
# Create attention mask from input_ids
attention_mask = (input_ids != pad_token_id).long()  # 1 for real, 0 for pad

# In attention function, expand and apply mask
mask = attention_mask.unsqueeze(1)  # [batch, seq_len] → [batch, 1, seq_len]
scores = scores.masked_fill(mask == 0, float('-inf'))

# After softmax, padded positions get ~0 weight
# Why -inf? Because exp(-inf) = 0, so softmax(-inf) = 0

# Verify masking works
max_attention_to_padding = weights[:, :real_length, real_length:].max()
assert max_attention_to_padding < 1e-6  # Should be ~0
```

### Text Tokenization (Basic)
```python
# Build vocabulary
from collections import Counter

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

# Encode text
def encode(text, vocab, max_len=100):
    tokens = text.split()[:max_len]
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return torch.tensor(ids)
```

---

## 🤗 HuggingFace Transformers

### Basic Usage
```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize
texts = ["Hello world", "PyTorch is great"]
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
# Returns: {'input_ids': ..., 'attention_mask': ...}

# Forward pass
outputs = model(**encoded)
logits = outputs.logits
```

### Fine-tuning with Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 📊 Evaluation Metrics

### Using torchmetrics
```python
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# Initialize
accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
f1 = F1Score(task='multiclass', num_classes=num_classes).to(device)

# Update with predictions
for batch in dataloader:
    preds = model(batch['input'])
    accuracy.update(preds, batch['label'])
    f1.update(preds, batch['label'])

# Compute
acc = accuracy.compute()
f1_score = f1.compute()

# Reset for next epoch
accuracy.reset()
f1.reset()
```

### Using sklearn
```python
from sklearn.metrics import classification_report, confusion_matrix

# Collect all predictions
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch['input'].to(device))
        preds = outputs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['label'].numpy())

# Compute metrics
print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
```

---

## 💡 Tips & Best Practices

### Debugging
```python
# Check for NaN
torch.isnan(tensor).any()

# Check tensor statistics
print(f'Mean: {tensor.mean()}, Std: {tensor.std()}, Min: {tensor.min()}, Max: {tensor.max()}')

# Gradient clipping (prevent exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: {param.grad.abs().mean()}')
```

### Mixed Precision Training (Faster on GPU)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Set Random Seeds
```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 🚀 Quick Reference

```python
# Model
model = MyModel()
model.to(device)
model.train()  # Training mode
model.eval()   # Evaluation mode

# Forward pass
outputs = model(inputs)

# Loss
loss = criterion(outputs, labels)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Evaluation
with torch.no_grad():
    outputs = model(inputs)

# Save/Load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```
