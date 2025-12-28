# Day 3: HuggingFace Without Crutches

**Time**: 4 hours | **File**: `bert_custom.py`

---

## Task 1: Fine-Tune BERT (NO Trainer) (90 min)

**Dataset**: SST-2 sentiment, small subset (2k train, 500 val)

### What to Build

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.cuda.amp as amp

# Load model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

class SST2Dataset(Dataset):
    def __getitem__(self, idx):
        # Tokenize and return {'input_ids', 'attention_mask', 'label'}
        pass

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    """
    Custom training loop with:
    - AdamW optimizer
    - Warmup scheduler
    - Mixed precision (AMP)
    """
    # Use torch.cuda.amp.autocast() for mixed precision
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    pass

def validate(model, dataloader, device):
    """Validation loop"""
    pass

# Main training
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps
)
scaler = amp.GradScaler()  # For mixed precision
```

### Confidence Check
✅ If `Trainer` disappeared tomorrow, I could still train models.

---

## Task 2: Attention Visualization (45 min)

### What to Build

```python
def extract_attention(model, text, tokenizer, device):
    """Extract attention weights from BERT"""
    # Forward with output_attentions=True
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True  # KEY
    )
    # outputs.attentions: tuple of (num_layers,)
    # Each: (batch, num_heads, seq_len, seq_len)
    return outputs.attentions, tokens

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """Plot attention heatmap"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    attn = attention_weights[layer][0, head].cpu().numpy()
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens)
    plt.title(f'Layer {layer}, Head {head}')
    plt.show()

# Test on examples
text = "This movie was absolutely fantastic!"
attentions, tokens = extract_attention(model, text, tokenizer, device)
visualize_attention(attentions, tokens, layer=0, head=0)
visualize_attention(attentions, tokens, layer=5, head=0)
```

**Analysis**:
- What does [CLS] attend to?
- Do different heads show different patterns?
- How does attention change across layers?

### Confidence Check
✅ I can explain what the model attends to and why.

---

## Task 3: Reflection (25 min)

**Save to**: `../reflections/day3_reflection.md`

**Answer (1 page)**:
1. What confused me on Day 1?
2. What confused me on Day 2?
3. What feels easy now?
4. What would I build next?

**Be thorough. This is the most important reflection.**

---

## Debugging Time (45 min)

**Experiments**:
1. Learning rates: [1e-5, 2e-5, 5e-5]
2. With vs without warmup
3. With vs without mixed precision (compare speed)
4. Visualize different layers and heads

---

## Expected Results

- Train acc: 90-95%
- Val acc: 85-90%
- Time: ~2-3 min/epoch on GPU
- Memory: ~2-3 GB GPU RAM

**If < 80% accuracy**:
- Check LR (try 2e-5)
- Ensure warmup is used
- Train for more epochs

---

## Final Confidence Checklist

After 3 days, you should be able to:

### Day 1
- [ ] Build training loops from scratch
- [ ] Debug NaNs and shape errors
- [ ] Handle tokenization and padding

### Day 2
- [ ] Derive attention math on paper
- [ ] Implement multi-head attention
- [ ] Build transformer encoder

### Day 3
- [ ] Fine-tune BERT without Trainer
- [ ] Use AdamW + warmup + AMP
- [ ] Visualize and interpret attention

**If you can do all this → You have real confidence.**

---

## What's Next?

1. Build a custom project on YOUR dataset
2. Implement GPT-style decoder
3. Deploy a model as API
4. Enter Kaggle competitions
5. Contribute to open source

**You're ready. Go build something! 🚀**
