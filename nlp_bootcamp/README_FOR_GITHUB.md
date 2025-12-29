# 3-Day PyTorch NLP & Transformers Bootcamp

**Optimization Target: CONFIDENCE (Not Accuracy)**

This is my personal 3-day intensive bootcamp to build real confidence in PyTorch, NLP, and Transformers.

## What I Built

### Day 1: I Own the Training Loop
- ✅ Text preprocessing pipeline from scratch
- ✅ Custom vocabulary with padding and masking
- ✅ Baseline NLP model (Embedding → Mean Pool → Linear)
- ✅ Complete training loop with gradient clipping
- ✅ Manual cross-entropy loss implementation
- ✅ NaN debugging experiments

**Dataset**: IMDb (5k samples)
**Result**: 75% accuracy
**Confidence**: Can change hyperparameters without panic ✓

### Day 2: Transformers Are Not Magic
- ✅ Scaled dot-product attention from scratch
- ✅ Multi-head attention mechanism
- ✅ Transformer encoder block
- ✅ Positional encoding
- ✅ Residual connections and layer normalization

**Dataset**: AG News (5k samples)
**Result**: 70% accuracy
**Confidence**: Can derive attention math on paper ✓

### Day 3: HuggingFace Without Crutches
- ✅ Fine-tuned BERT with custom training loop (NO Trainer API)
- ✅ AdamW optimizer with warmup scheduler
- ✅ Mixed precision training (AMP)
- ✅ Attention weight visualization
- ✅ Token-to-token attention analysis

**Dataset**: SST-2 (2k samples)
**Result**: 88% accuracy
**Confidence**: Can train transformers professionally ✓

## Key Learnings

1. **Training loops aren't magic** - Just forward, backward, step
2. **Attention is matrix multiplication** - QK^T/√d then softmax
3. **Debugging is about shapes** - Print shapes everywhere
4. **NaNs have specific causes** - Learning rate, no clipping, log(0)
5. **Transformers are understandable** - When you build them yourself

## Rules I Followed

- ⏱ 4 hours per day
- ❌ No Trainer APIs
- ❌ No copy-paste solutions
- ✅ Explain every line I wrote
- ✅ Mandatory daily reflections

## Tools Used

- PyTorch
- HuggingFace Transformers
- Jupyter Notebooks
- Python 3.12

## Time Investment

- Day 1: 4 hours (2.5 hrs coding, 1.5 hrs debugging)
- Day 2: 4 hours (hardest: attention mechanism)
- Day 3: 4 hours (easiest: building on Day 2 knowledge)

**Total**: 12 hours of focused learning

## What's Next

- [ ] Build a custom project with my own dataset
- [ ] Implement GPT-style decoder
- [ ] Fine-tune larger models (BERT-large, RoBERTa)
- [ ] Deploy a model as API

## Reflections

See individual notebook markdown cells for daily reflections.

---

**Bootcamp completed on**: [Your date]
**Confidence level**: 🔥🔥🔥🔥🔥
