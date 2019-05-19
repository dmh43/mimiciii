from random import shuffle

def get_cv_folds(num_folds, keys):
  num_keys = len(keys)
  fold_len = num_keys // num_folds
  permutation = list(keys)
  shuffle(permutation)
  first_folds = [permutation[i * fold_len : (i + 1) * fold_len] for i in range(num_folds - 1)]
  last_fold = permutation[(num_folds - 1) * fold_len:]
  return first_folds + [last_fold]

def deal_folds(coll, keys):
  test, train = [], []
  test_keys = set(keys)
  for key in coll.keys():
    if key in test_keys:
      test.append(coll[key])
    else:
      train.append(coll[key])
  return test, train
