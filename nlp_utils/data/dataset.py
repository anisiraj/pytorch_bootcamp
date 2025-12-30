"""Custom PyTorch Dataset for text classification."""

import torch
from torch.utils.data import Dataset


class ReviewDataSet(Dataset):
    """
    PyTorch Dataset for text classification tasks.

    Args:
        target: Dictionary or HuggingFace Dataset containing text and labels
        text_key: Key for text data (default: "text")
        label_key: Key for label data (default: "label")
        vocab: Vocabulary object for encoding text
        max_length: Maximum sequence length (default: 512)
    """

    def __init__(self, target, text_key="text", label_key="label", vocab=None, max_length=512):
        super().__init__()
        self.text_key, self.label_key = text_key, label_key
        self.vocab = vocab
        self.max_length = max_length

        # Validate input
        if all([key in target for key in (text_key, label_key)]):
            self.target = target
        else:
            raise ValueError(f"target must contain {self.text_key} and {self.label_key}")

    def __len__(self):
        return len(self.target[self.text_key])

    def __getitem__(self, index):
        tokens = self.vocab.encode(self.target[self.text_key][index])[:self.max_length]
        label = self.target[self.label_key][index]
        mask = [1] * len(tokens)

        return {
            'input_ids': tokens,
            'mask': mask,
            'label': label
        }


def collate_fn(batch):
    """
    Pad sequences in a batch to the same length.

    Args:
        batch: List of dicts from Dataset.__getitem__()
               Each dict has: {'input_ids': list, 'mask': list, 'label': int}

    Returns:
        Dict with padded tensors: {
            'input_ids': tensor [batch_size, max_len],
            'mask': tensor [batch_size, max_len],
            'label': tensor [batch_size]
        }
    """
    # Find the maximum sequence length in this batch
    max_len = max(len(item['input_ids']) for item in batch)

    # Initialize lists to collect padded sequences
    input_ids_padded = []
    masks_padded = []
    labels = []

    # Pad each item in the batch
    for item in batch:
        input_ids = item['input_ids']
        mask = item['mask']
        label = item['label']

        # Calculate how much padding is needed
        padding_len = max_len - len(input_ids)

        # Pad input_ids with 0 (PAD token index)
        padded_input_ids = input_ids + [0] * padding_len

        # Pad mask with 0 (means "ignore this position")
        padded_mask = mask + [0] * padding_len

        # Collect the padded sequences
        input_ids_padded.append(padded_input_ids)
        masks_padded.append(padded_mask)
        labels.append(label)

    # Convert lists to tensors
    return {
        'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
        'mask': torch.tensor(masks_padded, dtype=torch.float),
        'label': torch.tensor(labels, dtype=torch.long)
    }
