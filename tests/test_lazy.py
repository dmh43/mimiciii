from diag.lazy import Lazy, Cache


def test_cache(tmp_path):
  cache = Cache(tmp_path)
  assert not cache.is_cached('my_fn', {})

def test_cache_read(tmp_path):
  cache = Cache(tmp_path)
  cache.cache_result('my_fn', {}, 0)
  assert cache.is_cached('my_fn', {})
  assert cache.read_cache('my_fn', {}) == 0

# def test_lazy(tmp_path):
#   deferred = Lazy(int, tmp_path, cache)
