from collections import defaultdict
from pathlib import Path
import sqlite3
import pydash as _
import pickle

def realize(val):
  if isinstance(val, Lazy):
    return val.realize()
  else:
    return val

class Cache():
  def __init__(self, cache_path):
    self.cache_path = Path(cache_path)
    self.db_path = self.cache_path.joinpath('cache.db')
    self.connection = sqlite3.connect(self.db_path)
    self.cursor = self.connection.cursor()

  def _get_fn_cache_details(self, fn_name):
    self.cursor.execute(f'select * from cache_details where fn_name = {fn_name}')
    fn_cache_details = defaultdict(dict)
    for row in self.cursor.fetchall():
      fn_cache_details[row['cache_id']].update(row)
    return fn_cache_details

  def is_cached(self, fn_name, opts):
    fn_cache_details = self._get_fn_cache_details(fn_name)
    return _.some(fn_cache_details, lambda val, key: val == opts)

  def _path_from_details(self, cache_details):
    return self.cache_path.joinpath(str(cache_details['cache_id']))

  def read_cache(self, fn_name, opts):
    fn_cache_details = self._get_fn_cache_details(fn_name)
    cache_details = _.find(fn_cache_details, lambda val, key: val == opts)
    with open(self._path_from_details(cache_details), 'rb') as fh:
      return pickle.load(fh)

  def cache_result(self, fn_name, opts, result):
    fn_cache_details = self._get_fn_cache_details(fn_name)
    cache_details = _.find(fn_cache_details, lambda val, key: val == opts)
    with open(self._path_from_details(cache_details), 'rb') as fh:
      return pickle.dump(result, fh)


class LazyBuilder():
  def __init__(self, cache_path):
    self.cache_path = Path(cache_path)
    self.cache = Cache(self.cache_path)

  def __call__(self, fn):
    lazy = Lazy(fn, self.cache_path, self.cache)
    return lazy

class Lazy():
  def __init__(self, fn, cache_path, cache):
    self._fn, self._cache_path, self._cache = fn, Path(cache_path), cache
    self._args, self._kwargs, self._opts = None, None, None
    self._fn_name = self._fn.__name__

  def __call__(self, *args, opts=None, **kwargs):
    opts = opts if opts is not None else {}
    self._args, self._kwargs = args, kwargs
    self._opts = opts
    return self

  def _is_cached(self):
    return self._cache.is_cached(self._fn_name, self._opts)

  def _read_cache(self):
    return self._cache.read_cache(self._fn_name, self._opts)

  def _cache_result(self, result):
    return self._cache.cache_result(self._fn_name, self._opts, result)

  def realize(self):
    args = (realize(arg) for arg in self._args)
    kwargs = {key: realize(val) for key, val in self._kwargs.items()}
    if self._is_cached():
      return self._read_cache()
    else:
      result = self._fn(*args, opts=self._opts, **kwargs)
      self._cache_result(result)
      return result
