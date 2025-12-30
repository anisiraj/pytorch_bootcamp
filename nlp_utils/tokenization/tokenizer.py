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
