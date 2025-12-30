"""Data utilities for text classification."""

from nlp_utils.data.dataset import ReviewDataSet, collate_fn
from nlp_utils.data.utils import extract_imdb_sample_as_dict

__all__ = ['ReviewDataSet', 'collate_fn', 'extract_imdb_sample_as_dict']
