import torch
from torch.nn.utils.rnn import pad_sequence
from fastai.text import Tokenizer
import pydash as _

from operator import itemgetter
from itertools import combinations
from collections import Counter

pad_token_idx = 0
unk_token_idx = 1

def icd9_to_label(df):
  mapping = {}
  for icd9 in df.icd9_code.unique():
    mapping[icd9] = len(mapping)
  return mapping

def to_pairs_by_hadm_id(diagnoses_by_hadm_id):
  cnt = 0
  pairs_by_hadm_id = {}
  label_seq_nums_for_hadm_id = []
  current_hadm_id = None
  current_note_id = None
  for hadm_id, diag in diagnoses_by_hadm_id:
    hadm_id, note_id, seq_num, label = diag
    if (current_hadm_id is None) and (current_note_id is None):
      current_hadm_id = hadm_id
      current_note_id = note_id
    if current_hadm_id == hadm_id:
      label_seq_nums_for_hadm_id.append((label, seq_num))
    else:
      in_order = [label
                  for label, seq_num in sorted(label_seq_nums_for_hadm_id,
                                               key=itemgetter(1))]
      for pair in combinations(in_order, 2):
        pairs_by_hadm_id[hadm_id] = (current_note_id, pair)
        cnt += 1
      label_seq_nums_for_hadm_id = []
      label_seq_nums_for_hadm_id.append((label, seq_num))
      current_hadm_id = hadm_id
      current_note_id = note_id
  return pairs_by_hadm_id, cnt

def pad(batch, device=torch.device('cpu')):
  batch_lengths = torch.tensor(_.map_(batch, len),
                               dtype=torch.long,
                               device=device)
  return (pad_sequence(batch, batch_first=True, padding_value=1).to(device),
          batch_lengths)

def pad_to_len(coll, max_len, pad_with=None):
  pad_with = pad_with if pad_with is not None else pad_token_idx
  return coll + [pad_with] * (max_len - len(coll)) if len(coll) < max_len else coll

def collate_bow(bow):
  terms = []
  cnts = []
  max_len = 0
  for doc in bow:
    doc_terms = list(doc.keys())
    max_len = max(max_len, len(doc_terms))
    terms.append(doc_terms)
    cnts.append([doc[term] for term in doc_terms])
  terms = torch.tensor([pad_to_len(doc_terms, max_len) for doc_terms in terms])
  cnts = torch.tensor([pad_to_len(doc_term_cnts, max_len, pad_with=0) for doc_term_cnts in cnts])
  return terms, cnts

def tokens_to_indexes(tokens, lookup=None, num_tokens=None, token_set=None):
  def _append(coll, val, idx):
    if isinstance(coll, dict):
      coll[idx] = val
    else:
      coll.append(val)
  is_test = lookup is not None
  if lookup is None:
    lookup: dict = {'<unk>': unk_token_idx, '<pad>': pad_token_idx}
  result = []
  for idx, tokens_chunk in enumerate(tokens):
    tokens_to_parse = tokens_chunk if num_tokens is None else tokens_chunk[:num_tokens]
    chunk_result = []
    for token in tokens_to_parse:
      if (token_set is None) or (token in token_set):
        if is_test:
          chunk_result.append(lookup.get(token) or unk_token_idx)
        else:
          lookup[token] = lookup.get(token) or len(lookup)
          chunk_result.append(lookup[token])
      else:
        chunk_result.append(unk_token_idx)
    if len(chunk_result) > 0:
      _append(result, chunk_result, idx)
  return result, lookup

def get_default_tokenizer(): return Tokenizer()

def preprocess_texts(texts,
                     token_lookup=None,
                     num_tokens=None,
                     token_set=None,
                     get_tokenizer=get_default_tokenizer):
  tokenizer = get_tokenizer()
  tokenized = tokenizer.process_all(texts)
  idx_texts, token_lookup = tokens_to_indexes(tokenized,
                                              token_lookup,
                                              num_tokens=num_tokens,
                                              token_set=token_set)
  return idx_texts, token_lookup

def prepare_bow(lookup,
                token_lookup=None,
                token_set=None,
                num_tokens=None):
  ids = list(lookup.keys())
  contents = [lookup[text_id] for text_id in ids]
  numericalized, token_lookup = preprocess_texts(contents,
                                                 token_lookup=token_lookup,
                                                 token_set=token_set,
                                                 num_tokens=num_tokens)
  numericalized_bow = {text_id: Counter(doc)
                       for text_id, doc in zip(ids, numericalized)}
  return numericalized_bow, token_lookup
