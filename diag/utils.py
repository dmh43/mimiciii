from collections import Counter

import torch
import torch.nn as nn

class Identity(nn.Module):
  def forward(self, x): return x

def dont_update(module):
  for p in module.parameters():
    p.requires_grad = False

def do_update(module):
  for p in module.parameters():
    p.requires_grad = True

def append_at(obj, key, val):
  if key in obj:
    obj[key].append(val)
  else:
    obj[key] = [val]

def at_least_one_dim(tensor):
  if len(tensor.shape) == 0:
    return tensor.unsqueeze(0)
  else:
    return tensor

def to_list(coll):
  if isinstance(coll, torch.Tensor):
    return coll.tolist()
  else:
    return list(coll)

def maybe(val, default): return val if val is not None else default

def to_lookup(df, key_col_name, val_col_name):
  df_dict = df.to_dict('list')
  return dict(zip(df_dict[key_col_name], df_dict[val_col_name]))

def get_token_cnts(tokenizer, texts):
  tokenized = tokenizer.process_all(texts)
  cntr = Counter()
  for doc in tokenized: cntr.update(doc)
  return cntr
