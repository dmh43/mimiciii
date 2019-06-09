from collections import defaultdict
import uuid
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
    self.connection = sqlite3.connect(str(self.db_path))
    self.cursor = self.connection.cursor()
    with open('./sql/table_create.sql') as table_create_query_fh:
      query = table_create_query_fh.read()
      self.cursor.execute(query)

  def _get_fn_cache_details(self, fn_name):
    with open('./sql/get_cache_details.sql') as get_cache_details_query_fh:
      query = get_cache_details_query_fh.read().format(fn_name)
      self.cursor.execute(query)
    fn_cache_details = defaultdict(dict)
    for row in self.cursor.fetchall():
      fn_cache_details[row['cache_id']].update(row)
    return fn_cache_details

  def is_cached(self, fn_name, opts):
    fn_cache_details = self._get_fn_cache_details(fn_name)
    return _.some(fn_cache_details, lambda val, key: val == opts)

  def _path_from_cache_id(self, cache_id):
    return self.cache_path.joinpath(str(cache_id))

  def read_cache(self, fn_name, opts):
    fn_cache_details = self._get_fn_cache_details(fn_name)
    cache_details = _.find(fn_cache_details, lambda val, key: val == opts)
    with open(self._path_from_cache_id(cache_details), 'rb') as fh:
      return pickle.load(fh)

  def _create_cache_details(self, fn_name, opts):
    def _get_opt_cols(opt_name, opt_value):
      def _get_type(value):
        if isinstance(value, int): return 'int'
        elif isinstance(value, str): return 'str'
        elif isinstance(value, bool): return 'bool'
        else: raise ValueError
      return (opt_name, opt_value, _get_type(opt_value))
    cache_id = str(uuid.uuid1())
    opt_details_fields = ['opt_name', 'opt_value', 'opt_type']
    if len(opts) != 0:
      columns_str = ', '.join(['cache_id', 'fn_name'] + opt_details_fields)
      cache_details = [', '.join("'{}'".format(val)
                                 for val in [fn_name, cache_id] + [_get_opt_cols(opt_name, opt_value)
                                                                   for opt_name, opt_value in opts.items()])
                       for opt in opts]
    else:
      columns_str = ', '.join(['cache_id', 'fn_name'])
      cache_details = [', '.join("'{}'".format(val)
                                 for val in [fn_name, cache_id])]
    cache_details_str = ', '.join(['({})'.format(cache_detail)
                                   for cache_detail in cache_details])
    with open('./sql/create_cache_details.sql') as create_cache_details_fh:
      query = create_cache_details_fh.read().format(columns_str, cache_details_str)
      self.cursor.execute(query)
    return cache_id

  def cache_result(self, fn_name, opts, result):
    cache_id = self._create_cache_details(fn_name, opts)
    with open(self._path_from_cache_id(cache_id), 'wb') as fh:
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
