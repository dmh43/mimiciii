import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.optim import Adam
from pyrsistent import m
import pydash as _

from itertools import groupby
from functools import reduce
from operator import itemgetter

from diag.fetchers import fetch_note_by_id, fetch_icd_desc_lookup
from diag.preprocessing import icd9_to_label, to_pairs_by_hadm_id, get_default_tokenizer, prepare_bow, pad
from diag.utils import to_lookup, get_token_cnts
from diag.icd_encoder import ICDEncoder
from diag.note_encoder import NoteEncoder
from diag.pointwise_scorer import PointwiseScorer
from diag.pairwise_scorer import PairwiseScorer
from diag.pointwise_ranker import PointwiseRanker
from diag.dataset import MimicDataset, collate_fn
from diag.cv import get_cv_folds, deal_folds
from diag.metrics import metrics_at_k
from diag.ranking import to_rel_sets

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model_params = m(min_num_occurances=10,
                   token_embed_len=100)
  train_params = m(dropout_keep_prob=0.8,
                   use_layer_norm=True,
                   use_batch_norm=False,
                   use_bce_loss=False,
                   num_cv_folds=5)
  diagnoses_df = pd.read_csv('data/diagnoses.csv')
  num_icd9_codes = len(diagnoses_df.icd9_code.unique())
  label_lookup = icd9_to_label(diagnoses_df)
  diagnoses_df['label'] = diagnoses_df.icd9_code.map(lambda val: label_lookup[val])
  diagnoses_df.sort_values('hadm_id', inplace=True)
  diagnoses = diagnoses_df.to_dict('list')
  diagnoses_as_tuple = zip(diagnoses['hadm_id'],
                           diagnoses['note_id'],
                           diagnoses['seq_num'],
                           diagnoses['label'])
  diagnoses_by_hadm_id = groupby(diagnoses_as_tuple, itemgetter(0))
  pairs_by_hadm_id, num_pairs = to_pairs_by_hadm_id(diagnoses_by_hadm_id)
  notes_df = pd.read_csv('data/notes.csv')
  notes_lookup = to_lookup(notes_df, 'note_id', 'text')
  icd_desc_lookup = fetch_icd_desc_lookup()
  icd_desc_lookup_by_label = {label_lookup[icd9]: desc
                              for icd9, desc in icd_desc_lookup.items()}
  tokenizer = get_default_tokenizer()
  token_cnts = get_token_cnts(tokenizer, notes_lookup.values())
  icd_token_cnts = get_token_cnts(tokenizer, icd_desc_lookup_by_label.values())
  token_set = {token
               for token, cnt in token_cnts.items()
               if cnt + icd_token_cnts[token] >= model_params.min_num_occurances}
  token_set.update(icd_token_cnts.keys())
  notes_bow, token_lookup = prepare_bow(notes_lookup, token_set=token_set)
  icd_desc_bow, __ = prepare_bow(icd_desc_lookup_by_label, token_lookup=token_lookup, token_set=token_set)
  num_unique_tokens = len(token_lookup)
  folds = get_cv_folds(train_params.num_cv_folds, pairs_by_hadm_id.keys())
  for test_fold_num in range(train_params.num_cv_folds):
    print('Fold num', test_fold_num)
    test_keys = folds[test_fold_num]
    test, train = deal_folds(pairs_by_hadm_id, test_keys)
    test_rel_sets = to_rel_sets(test)
    test_hadm_ids = [hadm_id for hadm_id, g in groupby(test, itemgetter(0))]
    test_note_id_by_hadm_id = {hadm_id: g[1]
                               for hadm_id, g in groupby(test, itemgetter(0))}
    note = pad([notes_bow[test_note_id_by_hadm_id[hadm_id]] for hadm_id in test_hadm_ids],
               device=device)
    candidates = list(range(num_icd9_codes))
    icd = pad([icd_desc_bow[label] for label in candidates],
              device=device)
    token_embeds = nn.Embedding(num_unique_tokens, model_params.token_embed_len)
    pointwise_scorer = PointwiseScorer(token_embeds, token_embeds, model_params, train_params)
    pairwise_scorer = PairwiseScorer(token_embeds, token_embeds, model_params, train_params)
    ranker = PointwiseRanker(device, pointwise_scorer)
    criteria = nn.CrossEntropyLoss()
    optimizer = Adam(pairwise_scorer.parameters())
    dataset = MimicDataset(notes_bow, icd_desc_bow, train)
    dataloader = DataLoader(dataset, batch_sampler=BatchSampler(RandomSampler(dataset),
                                                                train_params.batch_size,
                                                                False))
    for batch_num, batch in enumerate(dataloader):
      optimizer.zero_grad()
      out = pairwise_scorer(*(tens.to(device) for tens in batch))
      loss = criteria(torch.zeros_like(out), out)
      if batch_num % 100 == 0: print('batch', batch_num, 'loss', loss)
      if batch_num % 10000 == 0:
        rankings = ranker(note, icd).tolist()
        print('batch', batch_num, metrics_at_k(rankings, test_rel_sets))
      loss.backward()
      optimizer.step()

if __name__ == "__main__": main()
