from collections import defaultdict
from itertools import groupby
from operator import itemgetter

def to_rel_sets(diags):
  rel = defaultdict(set)
  for hadm_id, diag in groupby(diags, itemgetter(0)):
    rel[hadm_id].add(list(diag)[-1])
  return dict(rel)
