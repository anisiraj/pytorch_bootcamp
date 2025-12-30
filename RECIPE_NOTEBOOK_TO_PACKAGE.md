# Recipe: Converting Jupyter Notebooks to Reusable Packages

A step-by-step guide to extracting reusable code from Jupyter notebooks into a proper Python package.

## Overview

This recipe shows how to transform exploratory code in Jupyter notebooks into a clean, reusable package that can be imported across multiple projects.

**Time**: ~30-45 minutes
**Tools**: `uv` for package management

---

## Step 1: Identify Reusable Components (5 min)

Review your notebook and identify code that:
- Is used multiple times across notebooks
- Contains general-purpose utilities
- Would benefit from being tested independently
- Clutters the notebook and distracts from the main workflow

### Example Components:
- ✅ Tokenization functions
- ✅ Vocabulary classes
- ✅ Custom Dataset classes
- ✅ Collate functions
- ✅ Data preprocessing utilities
- ✅ Helper functions for loading/sampling data

### Keep in Notebooks:
- ❌ Exploratory data analysis
- ❌ Model training loops (unless generalizable)
- ❌ Visualization code specific to one experiment
- ❌ One-off data transformations

---

## Step 2: Design Package Structure (5 min)

Create a logical directory structure based on functionality.

```bash
mkdir -p package_name/{module1,module2}
```

### Example Structure:
```
nlp_utils/
├── __init__.py              # Main package exports
├── tokenization/
│   ├── __init__.py
│   ├── tokenizer.py         # Tokenization functions
│   └── vocabulary.py        # Vocabulary class
└── data/
    ├── __init__.py
    ├── dataset.py           # Custom Dataset classes
    └── utils.py             # Helper functions
```

### Naming Guidelines:
- Use descriptive module names: `tokenization`, `data`, `models`
- Group related functionality together
- Keep file names singular: `tokenizer.py` not `tokenizers.py`
- Use `utils.py` for miscellaneous helpers

---

## Step 3: Extract Code from Notebook (10-15 min)

### 3.1 Copy Function/Class Definitions

Extract each component from the notebook into its own file:

```python
# nlp_utils/tokenization/tokenizer.py
"""Simple tokenizer for text classification."""

import re


def tokenize(text, lowercase=True, remove_punctuation=True, min_length=1):
    """
    Simple whitespace tokenizer for text classification.

    Args:
        text: Input text string
        lowercase: Convert to lowercase (default: True)
        remove_punctuation: Remove punctuation marks (default: True)
        min_length: Minimum token length to keep (default: 1)

    Returns:
        List of tokens
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)

    tokens = [token for token in text.split() if len(token) >= min_length]

    return tokens
```

### 3.2 Add Type Hints

Add proper type annotations for better IDE support:

```python
from typing import Callable
from collections import Counter

type Tokenizer = Callable[[str], list[str]]


class Vocabulary:
    def __init__(self, tokenizer: Tokenizer, min_freq: int = 1):
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, indices: list[int]) -> list[str]:
        ...
```

### 3.3 Improve Documentation

- Add module-level docstrings
- Document all parameters and return values
- Include usage examples in docstrings

---

## Step 4: Create `__init__.py` Files (5 min)

Make your package importable by creating `__init__.py` files.

### Module-level `__init__.py`:
```python
# nlp_utils/tokenization/__init__.py
"""Tokenization utilities."""

from nlp_utils.tokenization.tokenizer import tokenize
from nlp_utils.tokenization.vocabulary import Vocabulary

__all__ = ['tokenize', 'Vocabulary']
```

### Package-level `__init__.py`:
```python
# nlp_utils/__init__.py
"""NLP utilities for PyTorch practice."""

from nlp_utils.tokenization.tokenizer import tokenize
from nlp_utils.tokenization.vocabulary import Vocabulary
from nlp_utils.data.dataset import ReviewDataSet, collate_fn
from nlp_utils.data.utils import extract_imdb_sample_as_dict

__all__ = [
    'tokenize',
    'Vocabulary',
    'ReviewDataSet',
    'collate_fn',
    'extract_imdb_sample_as_dict',
]
```

**Key Points:**
- Export only public APIs in `__all__`
- Use absolute imports (`from nlp_utils.x` not `from .x`)
- Keep imports clean and organized

---

## Step 5: Configure `pyproject.toml` (3 min)

Add package configuration to tell `setuptools` which directories to include.

```toml
[project]
name = "your-project-name"
version = "0.1.0"
description = "Description of your project"
requires-python = ">=3.12"
dependencies = [
    # Your dependencies here
]

[tool.setuptools]
packages = ["nlp_utils"]  # List only your package, not notebooks
```

**Important:**
- Only include your package in `packages` list
- Exclude notebook directories to avoid conflicts
- Update description to reflect the package additions

---

## Step 6: Install Package in Editable Mode (2 min)

Install your package so changes are immediately reflected:

```bash
uv pip install -e .
```

**Benefits of editable mode:**
- ✅ No reinstall needed after code changes
- ✅ Import package from anywhere in your project
- ✅ Test changes immediately in notebooks

**Verify installation:**
```bash
python -c "from nlp_utils import tokenize, Vocabulary; print('Success!')"
```

---

## Step 7: Refactor Notebooks (10 min)

Replace notebook code with package imports.

### Before:
```python
# Cell 1: Import standard libraries
import torch
from torch.utils.data import Dataset

# Cell 2: Define tokenizer
def tokenize(text, lowercase=True, remove_punctuation=True):
    # ... 20 lines of code

# Cell 3: Define Vocabulary class
class Vocabulary:
    # ... 50 lines of code

# Cell 4: Define Dataset class
class ReviewDataSet(Dataset):
    # ... 30 lines of code

# Cell 5: Define collate function
def collate_fn(batch):
    # ... 25 lines of code

# Cell 6: Actually use the code
vocab = Vocabulary(tokenizer=tokenize)
# ...
```

### After:
```python
# Cell 1: Imports
import torch
from nlp_utils import tokenize, Vocabulary, ReviewDataSet, collate_fn

# Cell 2: Use the code directly
vocab = Vocabulary(tokenizer=tokenize)
# ...
```

**Result:**
- Notebook goes from ~125 lines to ~10 lines of imports
- Focus shifts to experimentation, not implementation
- Easier to read and understand

---

## Step 8: Document Your Package (5 min)

Create a README for your package with:

```markdown
# Package Name

Brief description.

## Installation

\`\`\`bash
uv pip install -e .
\`\`\`

## Usage

\`\`\`python
from nlp_utils import tokenize, Vocabulary

# Example usage
vocab = Vocabulary(tokenizer=tokenize)
vocab.build_from_texts(['hello world'])
\`\`\`

## Components

### tokenize()
Description and examples...

### Vocabulary
Description and examples...
```

Include:
- Installation instructions
- Basic usage examples
- API documentation for each component
- Package structure diagram

---

## Best Practices

### ✅ DO:
- **Use descriptive names**: `extract_imdb_sample_as_dict` not `sample_data`
- **Add type hints**: Makes code self-documenting and enables IDE support
- **Write docstrings**: Include args, returns, and examples
- **Keep functions focused**: One function = one responsibility
- **Group by functionality**: Related code in same module
- **Test your imports**: Verify package works before refactoring notebooks

### ❌ DON'T:
- **Create premature abstractions**: Extract code only when you'll reuse it
- **Over-engineer**: Keep it simple and focused
- **Skip documentation**: Future you will thank present you
- **Mix notebook and package code**: Keep clear separation
- **Forget to update `__init__.py`**: New modules need exports
- **Use generic names**: `utils.py` is OK, but `helpers.py` needs better naming

---

## Workflow Summary

```bash
# 1. Create package structure
mkdir -p package_name/{module1,module2}

# 2. Extract code from notebook into .py files
# (Copy and paste, add type hints, improve docs)

# 3. Create __init__.py files
touch package_name/__init__.py
touch package_name/module1/__init__.py

# 4. Configure pyproject.toml
# Add [tool.setuptools] packages = ["package_name"]

# 5. Install in editable mode
uv pip install -e .

# 6. Verify installation
python -c "from package_name import MyClass; print('Works!')"

# 7. Update notebooks to use imports
# Replace function definitions with: from package_name import ...

# 8. Document in README.md
```

---

## Example: Real-World Transformation

### Notebook Before (150 lines):
```python
# Lots of imports
import re
from collections import Counter
from torch.utils.data import Dataset
# ...

# Function definitions (80 lines)
def tokenize(...): ...
class Vocabulary: ...
class ReviewDataSet(Dataset): ...
def collate_fn(...): ...

# Actual work (70 lines)
ds = load_dataset("imdb")
vocab = Vocabulary(tokenizer=tokenize)
# ... model training
```

### Notebook After (70 lines):
```python
# Clean imports
from nlp_utils import tokenize, Vocabulary, ReviewDataSet, collate_fn

# Actual work (70 lines)
ds = load_dataset("imdb")
vocab = Vocabulary(tokenizer=tokenize)
# ... model training
```

### Package Structure Created:
```
nlp_utils/
├── __init__.py (14 lines)
├── tokenization/
│   ├── __init__.py (5 lines)
│   ├── tokenizer.py (27 lines)
│   └── vocabulary.py (73 lines)
└── data/
    ├── __init__.py (6 lines)
    ├── dataset.py (96 lines)
    └── utils.py (44 lines)
```

**Total package code**: ~265 lines (well-documented, reusable)
**Notebook reduction**: 150 → 70 lines (53% reduction)
**Reusability**: ∞ (can use across all future notebooks)

---

## Troubleshooting

### Import Errors
```python
ModuleNotFoundError: No module named 'nlp_utils'
```
**Solution**: Run `uv pip install -e .` from project root

### Multiple Top-Level Packages Error
```
setuptools error: Multiple top-level packages discovered
```
**Solution**: Add `[tool.setuptools] packages = ["your_package"]` to `pyproject.toml`

### Changes Not Reflected
**Solution**:
- Restart Jupyter kernel
- Verify you installed with `-e` flag (editable mode)
- Check you're importing from the right package

---

## Next Steps

Once you have a working package:

1. **Add tests**: Create `tests/` directory with pytest
2. **Add CI/CD**: Automate testing with GitHub Actions
3. **Version properly**: Follow semantic versioning
4. **Publish** (optional): Share on PyPI if generally useful
5. **Keep iterating**: Add new utilities as you identify patterns

---

## Summary

Converting notebooks to packages:
- ✅ Makes code reusable across projects
- ✅ Keeps notebooks clean and focused
- ✅ Enables better testing and documentation
- ✅ Improves code quality through abstraction
- ✅ Takes ~30-45 minutes for initial setup
- ✅ Pays dividends on every future notebook

**The best time to create a package is when you copy-paste code for the second time.**
