# 🧠 3-Day PyTorch NLP & Transformers Bootcamp

## **Optimization Target: CONFIDENCE (Not Accuracy)**

This bootcamp is designed to build *real confidence* in PyTorch, NLP
pipelines, and Transformers in **3 days (4 hours/day)**.\
The focus is **ownership, understanding, and debugging ability** --- not
benchmarks or leaderboards.

------------------------------------------------------------------------

## 🔒 Constraints (Non‑Negotiable)

-   ⏱ 4 hours per day
-   🚀 GPU available
-   ❌ No `Trainer` APIs
-   ❌ No copy‑paste full solutions
-   ✅ You must be able to explain every line you write

------------------------------------------------------------------------

## ⏰ Daily Structure

-   **20 min** --- Concept framing & planning\
-   **2.5 hrs** --- Coding from a blank file\
-   **45 min** --- Debugging & experiments\
-   **25 min** --- Written reflection

------------------------------------------------------------------------

# 🔥 DAY 1 --- *I Own the Training Loop*

### 🎯 Goal

Train an NLP model **from scratch** and debug it confidently.

------------------------------------------------------------------------

### Task 1 --- Text Pipeline

**Dataset:** IMDb (5k samples)

Build: - Whitespace tokenizer\
- Vocabulary (`PAD=0`, `UNK=1`)\
- Padding and attention masks\
- Custom `Dataset` and `DataLoader`

**Confidence Check** \> I can explain why incorrect padding or masking
breaks training.

------------------------------------------------------------------------

### Task 2 --- Baseline NLP Model

**Model**

    Embedding → Mean Pool → Linear

You must: - Write your own training + validation loop - Move model &
data to GPU - Add gradient clipping

**Confidence Check** \> I can change embedding size or max sequence
length without panic.

------------------------------------------------------------------------

### Task 3 --- Loss Mastery

Build: - Cross‑Entropy Loss manually - Compare with
`nn.CrossEntropyLoss` - Intentionally create NaNs and fix them

**Confidence Check** \> I know exactly where NaNs come from.

------------------------------------------------------------------------

# 🔥 DAY 2 --- *Transformers Are Not Magic*

### 🎯 Goal

Understand and implement attention **mechanistically**.

------------------------------------------------------------------------

### Task 1 --- Scaled Dot‑Product Attention

Build: - Q, K, V projections - Attention score matrix - Scaling by √d -
Proper masking

**Confidence Check** \> I can derive attention math on paper.

------------------------------------------------------------------------

### Task 2 --- Multi‑Head Attention

Build: - Split embeddings into heads - Apply attention per head -
Concatenate and project output

**Confidence Check** \> I can explain why multiple heads help.

------------------------------------------------------------------------

### Task 3 --- Transformer Encoder Block

**Dataset:** AG News (5k samples)

Build: - Residual connections - Layer normalization - Feed‑forward
network

Goal: - Training runs - Loss decreases

**Confidence Check** \> I can remove positional encoding and explain
what breaks.

------------------------------------------------------------------------

# 🔥 DAY 3 --- *HuggingFace Without Crutches*

### 🎯 Goal

Use transformers **professionally**, not blindly.

------------------------------------------------------------------------

### Task 1 --- Fine‑Tune BERT (No Trainer)

**Dataset:** SST‑2 (small subset)

Build: - Load model & tokenizer - Custom training loop - AdamW
optimizer - LR scheduler - Mixed precision (AMP)

**Confidence Check** \> If `Trainer` disappeared tomorrow, I could still
train models.

------------------------------------------------------------------------

### Task 2 --- Attention Visualization

Build: - Extract attention weights - Visualize token‑to‑token attention

**Confidence Check** \> I can explain what the model attends to and why.

------------------------------------------------------------------------

### Task 3 --- Reflection (Mandatory)

Write one page answering: 1. What confused me on Day 1? 2. What confused
me on Day 2? 3. What feels easy now? 4. What would I build next?

------------------------------------------------------------------------

## 🏁 Final Outcome

After these 3 days: - You can start PyTorch NLP projects **without
fear** - You understand transformers **deeply** - You can debug shape,
loss, and training issues - You trust yourself again

That is **real confidence**.
