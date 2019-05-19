import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from pyrsistent import m

from diag.fetchers import fetch_note_by_id, fetch_icd_desc_lookup
from diag.preprocessing import icd9_to_label, to_pairs, get_default_tokenizer, prepare_bow
from diag.utils import to_lookup, get_token_cnts
from diag.icd_encoder import ICDEncoder
from diag.note_encoder import NoteEncoder
from diag.pointwise_scorer import PointwiseScorer
from diag.pairwise_scorer import PairwiseScorer
from diag.pointwise_ranker import PointwiseRanker
from diag.dataset import MimicDataset

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model_params = m(min_num_occurances=10,
                   token_embed_len=100)
  train_params = m(dropout_keep_prob=0.8,
                   use_layer_norm=True,
                   use_batch_norm=False,
                   use_bce_loss=False)
  diagnoses_df = pd.read_csv('data/diagnoses.csv')
  label_lookup = icd9_to_label(diagnoses_df)
  diagnoses_df['label'] = diagnoses_df.icd9_code.map(lambda val: label_lookup[val])
  diagnoses_df.sort_values('hadm_id', inplace=True)
  diagnoses = diagnoses_df.to_dict('list')
  training_pairs = to_pairs(diagnoses)
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
  icd_desc_bow, __ = prepare_bow(notes_lookup, token_lookup=token_lookup, token_set=token_set)
  num_unique_tokens = len(token_lookup)
  token_embeds = nn.Embedding(num_unique_tokens, model_params.token_embed_len)
  pointwise_scorer = PointwiseScorer(token_embeds, token_embeds, model_params, train_params)
  pairwise_scorer = PairwiseScorer(token_embeds, token_embeds, model_params, train_params)
  ranker = PointwiseRanker(device, pointwise_scorer)
  dataset = MimicDataset(notes_bow, icd_desc_bow, training_pairs)
  dataloader = DataLoader(dataset, batch_sampler=BatchSampler(RandomSampler(), train_params.batch_size, False))




if __name__ == "__main__": main()
