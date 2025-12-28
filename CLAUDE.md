# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch practice and tutorial repository for learning deep learning with PyTorch and related libraries. The project uses `uv` for dependency management with Python 3.12.

## Environment Setup

This project uses `uv` for package management. The virtual environment is located at `.venv`.

### Install Dependencies
```bash
uv sync
```

### Activate Virtual Environment
```bash
source .venv/bin/activate
```

## Key Dependencies

The project includes the following major frameworks and libraries:

- **PyTorch ecosystem**: `torch`, `torchvision`, `torchmetrics`
- **Deep learning frameworks**:
  - `lightning` (PyTorch Lightning for structured training)
  - `fastai` (high-level deep learning library)
  - `transformers` (Hugging Face for NLP models)
  - `diffusers` (Hugging Face for diffusion models)
- **Scientific computing**: `numpy`, `pandas`, `scikit-learn`, `scikit-image`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`
- **Computer vision**: `opencv-python`
- **Optimization**: `optuna` (hyperparameter tuning)
- **Utilities**: `tqdm`, `ipykernel` (Jupyter support)

## Running Jupyter Notebooks

Since `ipykernel` is installed, Jupyter notebooks can be used for interactive development:

```bash
jupyter notebook
# or
jupyter lab
```

The Python kernel from `.venv` will be available for notebooks.

## Development Workflow

This repository is for learning and experimentation. When creating new notebooks or scripts:

1. Place notebooks or Python scripts in logical directories (e.g., `tutorials/`, `experiments/`, `models/`)
2. Use descriptive names that indicate the topic or concept being practiced
3. For PyTorch Lightning projects, follow the standard structure with separate files for models, data modules, and training scripts
4. For computer vision tasks, store datasets in a `data/` directory (add to `.gitignore` if large)

## Common Patterns

### PyTorch Lightning Structure
When using PyTorch Lightning, follow this organization:
- `LightningModule` subclasses define models with `training_step`, `validation_step`, etc.
- `LightningDataModule` subclasses handle data loading
- Training scripts use `Trainer` with callbacks

### fastai Structure
When using fastai, use `DataLoaders` and `Learner` abstractions with the high-level training API.

### Transformers
For Hugging Face transformers, use `AutoModel`, `AutoTokenizer` patterns and the `Trainer` API for fine-tuning.
