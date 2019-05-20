from pathlib import Path

def realize(val):
  if isinstance(val, Lazy):
    return val.realize()
  else:
    return val

class Cache():
  def __init__(self, cache_path):
    self.cache_path = Path(cache_path)

  def is_cached(self, fn_name, opts):
    pass

  def read_cache(self, fn_name, opts):
    pass

  def cache_result(self, fn_name, opts, result):
    pass


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
