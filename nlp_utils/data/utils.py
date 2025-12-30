"""Utility functions for data processing."""

from datasets import DatasetDict


def extract_imdb_sample_as_dict(
    dataset: DatasetDict,
    train_size: int = 5000,
    test_size: int = 1000,
    seed: int = 42,
    shuffle: bool = True
) -> tuple[dict, dict]:
    """
    Extract sampled train and test splits from IMDb dataset as dictionaries.

    Args:
        dataset: HuggingFace DatasetDict with 'train' and 'test' splits
        train_size: Number of samples for training (default: 5000)
        test_size: Number of samples for testing (default: 1000)
        seed: Random seed for shuffling (default: 42)
        shuffle: Whether to shuffle before sampling (default: True)

    Returns:
        Tuple of (train_dict, test_dict) where each is a dict with lists
        Each dict has keys like 'text' and 'label' with lists as values

    Example:
        >>> from datasets import load_dataset
        >>> ds = load_dataset("stanfordnlp/imdb")
        >>> train, test = extract_imdb_sample_as_dict(ds, train_size=5000, test_size=1000)
        >>> print(type(train))  # <class 'dict'>
        >>> print(train.keys())  # dict_keys(['text', 'label'])
    """
    train_ds = dataset['train']
    test_ds = dataset['test']

    if shuffle:
        train_ds = train_ds.shuffle(seed=seed)
        test_ds = test_ds.shuffle(seed=seed)

    train = train_ds.select(range(train_size))[:]
    test = test_ds.select(range(test_size))[:]

    return train, test