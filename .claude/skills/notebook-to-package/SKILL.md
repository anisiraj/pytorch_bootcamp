---
name: notebook-to-package
description: Convert Jupyter notebook code into a reusable Python package. Use when the user wants to refactor notebooks, extract reusable code, create a package from notebook functions, or organize notebook utilities. Triggers on phrases like "convert notebook to package", "refactor notebook", "extract code to package", "make this reusable", or "create package from notebook".
---

# Notebook to Package Converter

Expert skill for transforming Jupyter notebook code into clean, reusable Python packages.

## When to Use This Skill

- User wants to extract reusable functions/classes from notebooks
- Multiple notebooks share similar code
- Code needs to be tested and versioned separately
- User wants to organize utilities into a package
- Converting exploratory code into production-ready modules

## Workflow

### Step 1: Analyze the Notebook

1. Ask the user for the notebook path if not provided
2. Read and analyze the notebook contents
3. Identify reusable components:
   - Functions (especially utility functions)
   - Classes (Dataset, Vocabulary, custom models)
   - Collate functions
   - Data preprocessing utilities
   - Tokenizers and text processors
4. Present findings to user and ask which components to extract
5. Suggest a package name based on the functionality

### Step 2: Design Package Structure

1. Propose a logical directory structure based on functionality:
   ```
   package_name/
   ├── __init__.py
   ├── module1/
   │   ├── __init__.py
   │   └── component.py
   └── module2/
       ├── __init__.py
       └── utils.py
   ```

2. Grouping guidelines:
   - `tokenization/` - Tokenizers, vocabulary, text processing
   - `data/` - Dataset classes, collate functions, data utils
   - `models/` - Model architectures (if generalizable)
   - `utils/` - Miscellaneous helpers

3. Get user approval on structure

### Step 3: Extract and Enhance Code

For each component:

1. Create appropriate `.py` file in the package structure
2. Copy the code from notebook
3. **Add type hints** to all functions and classes:
   ```python
   def extract_sample(dataset: DatasetDict, size: int = 1000) -> dict:
       ...
   ```
4. **Write comprehensive docstrings**:
   ```python
   """
   Brief description of what the function does.

   Args:
       param1: Description with type info
       param2: Description with type info

   Returns:
       Description of return value

   Example:
       >>> from package import function
       >>> result = function(arg1, arg2)
       >>> print(result)
   """
   ```
5. Add module-level docstrings at top of each file
6. Organize imports properly (stdlib, third-party, local)

### Step 4: Create Package Infrastructure

1. **Create `__init__.py` for each module**:
   ```python
   """Module description."""

   from package.module.component import MyClass, my_function

   __all__ = ['MyClass', 'my_function']
   ```

2. **Create main package `__init__.py`**:
   ```python
   """Package description."""

   from package.module1.component import Class1
   from package.module2.utils import function1

   __all__ = [
       'Class1',
       'function1',
   ]
   ```

3. **Important**: Use absolute imports, not relative:
   - ✅ `from package.module import Class`
   - ❌ `from .module import Class`

### Step 5: Configure pyproject.toml

1. Read existing `pyproject.toml`
2. Add or update `[tool.setuptools]` section:
   ```toml
   [tool.setuptools]
   packages = ["package_name"]
   ```
3. **Critical**: Only list the new package, exclude notebook directories
4. Update project description if needed

### Step 6: Install and Verify

1. Run installation:
   ```bash
   uv pip install -e .
   ```

2. Verify with test import:
   ```bash
   python -c "from package_name import Component; print('✓ Success')"
   ```

3. Handle errors:
   - **ModuleNotFoundError**: Re-run `uv pip install -e .`
   - **Multiple packages error**: Fix `pyproject.toml` to exclude notebooks
   - **Import errors**: Check `__init__.py` files and import paths

### Step 7: Refactor Notebook

1. Add imports at the top:
   ```python
   from package_name import Class1, Class2, function1
   ```

2. Delete old function/class definitions from notebook

3. Keep notebook focused on:
   - Loading data
   - Running experiments
   - Training models
   - Visualizations

4. Add comments explaining the cleaner structure

### Step 8: Document the Package

Create or update `package_name/README.md`:

```markdown
# Package Name

Brief description

## Installation

\`\`\`bash
uv pip install -e .
\`\`\`

## Usage

\`\`\`python
from package_name import Component

# Example usage
component = Component()
\`\`\`

## Components

### Component1
Description and examples...

## Development

This package is installed in editable mode. Changes to source files
are immediately available without reinstalling.
```

## Best Practices

### Code Quality

- **Type hints on everything**: Functions, class attributes, return types
- **Descriptive names**: `extract_imdb_sample_as_dict` not `sample_data`
- **Single responsibility**: One function does one thing well
- **Clear docstrings**: Args, Returns, Examples
- **Module docstrings**: At top of every `.py` file

### Naming Conventions

- **Packages**: `lowercase_with_underscores`
- **Modules**: `singular.py` (e.g., `tokenizer.py` not `tokenizers.py`)
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_CASE`

### Import Organization

```python
# Standard library
import re
from typing import Callable

# Third-party
import torch
from datasets import DatasetDict

# Local package
from package.module import helper
```

### What to Extract

✅ **DO extract**:
- Reusable functions and utilities
- Custom Dataset/DataLoader classes
- Tokenizers and vocabulary classes
- Data preprocessing functions
- Generalizable model architectures
- Collate functions
- Helper functions used multiple times

❌ **DON'T extract**:
- Exploratory data analysis
- One-off visualizations
- Experiment-specific training loops
- Quick debugging code
- Notebook-specific configurations

## Error Handling

### Import Errors
```
ModuleNotFoundError: No module named 'package'
```
**Fix**: Ensure `uv pip install -e .` was run from project root

### Multiple Packages Error
```
Multiple top-level packages discovered in a flat-layout
```
**Fix**: Add `[tool.setuptools] packages = ["your_package"]` to `pyproject.toml`

### Changes Not Reflected
**Fix**:
- Restart Jupyter kernel
- Verify installed with `-e` flag
- Check importing from correct package

### Import Path Errors
**Fix**:
- Verify all `__init__.py` files exist
- Use absolute imports in package code
- Check `__all__` matches actual exports

## Interactive Approach

Always:
1. Ask clarifying questions before major steps
2. Get user approval on proposed structure
3. Show what you're about to create
4. Verify at each stage (installation, imports, etc.)
5. Provide clear before/after comparisons

## Output Summary

At the end, provide:

```
✅ Package Structure Created:
   package_name/
   ├── __init__.py (X lines)
   ├── module1/ (Y lines)
   └── module2/ (Z lines)

✅ Notebook Reduction:
   Before: 150 lines
   After: 70 lines (53% reduction)

✅ Import Statement:
   from package_name import Component1, Component2

✅ Verification:
   ✓ Package installed successfully
   ✓ All imports working
   ✓ Ready to use in notebooks

📝 Next Steps:
   - Add tests in tests/ directory
   - Set up CI/CD with GitHub Actions
   - Consider publishing to PyPI
```

## Tips for Success

- Keep `SKILL.md` focused, extract details to supporting files
- Always verify installation before refactoring notebooks
- Use descriptive commit messages when committing package code
- Document as you go, don't save it for the end
- Test imports immediately after creating each component
- Get user confirmation before major refactorings
