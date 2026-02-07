# Tests for nlp_utils Package

This directory contains comprehensive tests for the `nlp_utils` package.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_pooling.py
```

### Run with verbose output
```bash
pytest tests/test_pooling.py -v
```

### Run specific test class or method
```bash
pytest tests/test_pooling.py::TestMaskedMeanPool::test_basic_pooling -v
```

## Test Coverage

### Pooling Tests (`test_pooling.py`)

The pooling tests cover:

- **Basic pooling**: Standard mean pooling without padding
- **Padding handling**: Correct masking of padding tokens
- **Edge cases**: All-padding sequences, single tokens, varying lengths
- **Gradient flow**: Ensures backpropagation works correctly
- **Different batch sizes**: From 1 to large batches
- **Large dimensions**: BERT-like embedding dimensions (768+)
- **Soft masking**: Float-valued masks for weighted pooling

## Example Usage

See the `examples/pooling_demo.py` file for practical demonstrations of using the pooling layers:

```bash
python examples/pooling_demo.py
```

## Writing New Tests

When adding new components to `nlp_utils`, follow this structure:

1. Create a test file: `test_<component_name>.py`
2. Use pytest fixtures for common setup
3. Test normal cases, edge cases, and error conditions
4. Verify gradient flow for differentiable components
5. Test with realistic dimensions and batch sizes

Example test structure:

```python
import pytest
import torch
from nlp_utils.models.your_module import YourClass


class TestYourClass:
    """Test cases for YourClass."""

    def test_basic_functionality(self):
        """Test basic usage."""
        obj = YourClass()
        result = obj(input_data)
        assert result.shape == expected_shape

    def test_edge_case(self):
        """Test edge case."""
        # Your test here
        pass
```
