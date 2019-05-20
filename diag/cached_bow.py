from collections import Counter
import pickle
from pathlib import Path

from .preprocessing import tokens_to_indexes

class CachedBoW():
  def __init__(self, tokenizer, path, token_lookup):
    self.tokenizer, self.path, self.token_lookup = tokenizer, Path(path), token_lookup
    self.cache_path = Path('./cache/').joinpath(self.path)

  def _get_bow_name(self, key): return str(key)

  def _cached_get(self, key):
    with open(self.cache_path.joinpath(self._get_bow_name(key)), 'rb') as fh:
      return pickle.load(fh)

  def _cache_result(self, key, bow):
    with open(self.cache_path.joinpath(self._get_bow_name(key)), 'wb') as fh:
      pickle.dump(fh, bow)

  def _get_bow(self, key):
    with open(self.path.joinpath(self._get_bow_name(key))) as fh:
      text = fh.read()
      tokenized = self.tokenizer.process_text(text, self.tokenizer.tok_func(self.tokenizer.lang))
      numericalized, __ = tokens_to_indexes([tokenized], self.token_lookup)
      return dict(Counter(numericalized[0]))

  def __getitem__(self, key):
    try:
      bow = self._cached_get(key)
    except FileNotFoundError:
      bow = self._get_bow(key)
      self._cache_result(key, bow)
    return bow
