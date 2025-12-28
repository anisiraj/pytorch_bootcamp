# 🧠 3-Day PyTorch NLP & Transformers Bootcamp

## **Optimization Target: CONFIDENCE (Not Accuracy)**

This bootcamp is designed to build *real confidence* in PyTorch, NLP pipelines, and Transformers in **3 days (4 hours/day)**.

The focus is **ownership, understanding, and debugging ability** — not benchmarks or leaderboards.

---

## 🔒 Constraints (Non-Negotiable)

- ⏱ **4 hours per day**
- 🚀 **GPU available**
- ❌ **No `Trainer` APIs**
- ❌ **No copy-paste full solutions**
- ✅ **You must be able to explain every line you write**

---

## 📓 Format: Jupyter Notebooks

**Why notebooks?**
- ✅ Fast iteration and debugging
- ✅ Visual feedback for shapes and outputs
- ✅ Inline plots for attention visualization
- ✅ Easy experimentation
- ✅ Keep your work and experiments

**Rule**: Work top-to-bottom. No skipping around. Build systematically.

---

## ⏰ Daily Structure

- **20 min** — Concept framing & planning (read requirements)
- **2.5 hrs** — Coding in notebook from scratch
- **45 min** — Debugging & experiments
- **25 min** — Written reflection (markdown cells at end)

---

# 🔥 DAY 1 — *I Own the Training Loop*

**File**: `day1_training_loop.ipynb`

### 🎯 Goal
Train an NLP model **from scratch** and debug it confidently.

---

### Task 1 — Text Pipeline

**Dataset:** IMDb (5k samples)

**Build:**
- Whitespace tokenizer
- Vocabulary (`PAD=0`, `UNK=1`)
- Padding and attention masks
- Custom `Dataset` and `DataLoader`

**Confidence Check**
> I can explain why incorrect padding or masking breaks training.

---

### Task 2 — Baseline NLP Model

**Model:**
```
Embedding → Mean Pool → Linear
```

**You must:**
- Write your own training + validation loop
- Move model & data to GPU
- Add gradient clipping

**Confidence Check**
> I can change embedding size or max sequence length without panic.

---

### Task 3 — Loss Mastery

**Build:**
- Cross-Entropy Loss manually
- Compare with `nn.CrossEntropyLoss`
- Intentionally create NaNs and fix them

**Confidence Check**
> I know exactly where NaNs come from.

---

# 🔥 DAY 2 — *Transformers Are Not Magic*

**File**: `day2_attention.ipynb`

### 🎯 Goal
Understand and implement attention **mechanistically**.

---

### Task 1 — Scaled Dot-Product Attention

**Build:**
- Q, K, V projections
- Attention score matrix
- Scaling by √d
- Proper masking

**Confidence Check**
> I can derive attention math on paper.

---

### Task 2 — Multi-Head Attention

**Build:**
- Split embeddings into heads
- Apply attention per head
- Concatenate and project output

**Confidence Check**
> I can explain why multiple heads help.

---

### Task 3 — Transformer Encoder Block

**Dataset:** AG News (5k samples)

**Build:**
- Residual connections
- Layer normalization
- Feed-forward network

**Goal:**
- Training runs
- Loss decreases

**Confidence Check**
> I can remove positional encoding and explain what breaks.

---

# 🔥 DAY 3 — *HuggingFace Without Crutches*

**File**: `day3_bert_custom.ipynb`

### 🎯 Goal
Use transformers **professionally**, not blindly.

---

### Task 1 — Fine-Tune BERT (No Trainer)

**Dataset:** SST-2 (small subset)

**Build:**
- Load model & tokenizer
- Custom training loop
- AdamW optimizer
- LR scheduler
- Mixed precision (AMP)

**Confidence Check**
> If `Trainer` disappeared tomorrow, I could still train models.

---

### Task 2 — Attention Visualization

**Build:**
- Extract attention weights
- Visualize token-to-token attention (inline heatmaps)

**Confidence Check**
> I can explain what the model attends to and why.

---

### Task 3 — Reflection (Mandatory)

**In markdown cells at end of notebook, write:**
1. What confused me on Day 1?
2. What confused me on Day 2?
3. What feels easy now?
4. What would I build next?

---

## 🏁 Final Outcome

After these 3 days:
- You can start PyTorch NLP projects **without fear**
- You understand transformers **deeply**
- You can debug shape, loss, and training issues
- You trust yourself again

**That is real confidence.**

---

## 🚀 Getting Started

```bash
cd nlp_bootcamp

# Start Jupyter
jupyter notebook
# or
jupyter lab

# Create Day 1 notebook
# File → New → Notebook
# Name it: day1_training_loop.ipynb

# Start coding!
```

---

## 📋 Notebook Structure (Recommended)

### Day 1 Notebook Structure:
```
# Cell 1: Imports and Setup
# Cell 2: Load Data (IMDb 5k)
# Cell 3: Tokenization Function
# Cell 4: Vocabulary Class
# Cell 5: Test Vocabulary
# Cell 6: Dataset Class
# Cell 7: Test Dataset
# Cell 8: Model Architecture
# Cell 9: Test Model Forward Pass
# Cell 10: Training Loop
# Cell 11: Validation Loop
# Cell 12: Main Training
# Cell 13: Manual Cross-Entropy
# Cell 14: NaN Experiments
# Cell 15: Debugging Experiments
# Cell 16: Reflection (Markdown)
```

---

## 📝 Resources

- **PyTorch Cheatsheet**: `PYTORCH_CHEATSHEET.md`
- **Requirements**: Each day has `REQUIREMENTS.md` with details
- **Your determination**

---

## 💡 Notebook Tips

### DO:
- ✅ Work top-to-bottom
- ✅ Test each function immediately after writing
- ✅ Print shapes constantly
- ✅ Visualize outputs inline
- ✅ Keep markdown notes between cells

### DON'T:
- ❌ Jump around cells randomly
- ❌ Restart kernel constantly
- ❌ Keep old buggy code in cells
- ❌ Run cells out of order

---

## 🎯 Why This Works

**Scripts**: Run → Wait → See if it works → Debug → Repeat
**Notebooks**: Write → Run cell → Immediate feedback → Iterate

**For learning**: Notebooks are faster, more visual, more engaging.

**For production**: You'll convert to scripts later. But for learning? Notebooks win.

---

Ready to start? Open Jupyter and create `day1_training_loop.ipynb`! 🚀
