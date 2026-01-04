# TODO: Future Improvements for nlp_utils

## ✅ Completed

### ~~Merge Tokenizer and Vocabulary into Single Class~~ (DONE - 2024-12-30)

**Implemented Solution**:
- Created `SimpleTokenizer` class combining tokenization + vocabulary
- Defined `TokenizerProtocol` for interface compatibility
- Updated `SentimentModel` and `ReviewDataSet` to accept any tokenizer implementing the protocol
- Now compatible with HuggingFace tokenizers and custom implementations
- Removed old `Vocabulary` and `tokenize` from main API

**Problem**: ~~Current design has tight coupling between `tokenize` function and `Vocabulary` class~~ RESOLVED
- User must manually pass tokenizer to vocabulary
- Can't save/load complete tokenization state
- Doesn't match industry standards (HuggingFace, spaCy)

**Current Usage**:
```python
vocab = Vocabulary(tokenizer=tokenize)
vocab.build_from_texts(texts)
ids = vocab.encode(text)
```

**Proposed Design**:
```python
class Tokenizer:
    """Combined tokenization and vocabulary management."""

    def __init__(self, lowercase=True, remove_punctuation=True, min_freq=1):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_freq = min_freq
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.idx2token = {0: '<PAD>', 1: '<UNK>'}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        pass

    def build_vocab(self, texts: list[str]):
        """Build vocabulary from collection of texts."""
        pass

    def encode(self, text: str) -> list[int]:
        """Tokenize and encode text to IDs."""
        pass

    def decode(self, ids: list[int]) -> list[str]:
        """Decode IDs back to tokens."""
        pass

    def save(self, path: str):
        """Save tokenizer state to file."""
        pass

    def load(self, path: str):
        """Load tokenizer state from file."""
        pass

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1
```

**New Usage**:
```python
tokenizer = Tokenizer()
tokenizer.build_vocab(texts)
ids = tokenizer.encode(text)
```

**Benefits**:
- ✅ Single source of truth
- ✅ Easier to save/load complete state
- ✅ Cleaner API - no passing functions
- ✅ Matches HuggingFace pattern
- ✅ Self-contained and intuitive

**Migration Path**:
1. Create new `Tokenizer` class in `tokenization/tokenizer.py`
2. Keep old `Vocabulary` and `tokenize` for backwards compatibility
3. Update docs to recommend new `Tokenizer`
4. Add deprecation warnings after one version
5. Remove old API after two versions

## Medium Priority

### Add Model Checkpointing to Trainer
- Save best model weights automatically
- Support loading checkpoints to resume training
- Add early stopping based on validation metrics

### Support for Custom Tokenization Strategies
- BPE (Byte-Pair Encoding)
- WordPiece
- SentencePiece integration
- Character-level tokenization

### Add More Metrics
- Confusion matrix support
- Per-class metrics
- Custom metric callbacks

## Low Priority

### Add Data Augmentation
- Random token deletion
- Synonym replacement
- Back-translation

### Support for Multi-task Learning
- Multiple loss functions
- Task-specific heads
- Shared encoder architecture

### Add Visualization Utilities
- Plot training curves
- Attention visualization
- Embedding visualization with t-SNE/UMAP

---

## Notes

**Design Philosophy**:
- Follow HuggingFace/spaCy patterns where applicable
- Keep API simple and intuitive
- Maintain backwards compatibility with deprecation warnings
- Document migration paths clearly
