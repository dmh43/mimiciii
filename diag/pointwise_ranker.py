import pydash as _

import torch
import torch.nn as nn

from .preprocessing import collate_bow
from .utils import at_least_one_dim

class PointwiseRanker:
  def __init__(self,
               device,
               pointwise_scorer,
               chunk_size=-1):
    self.device = device
    self.pointwise_scorer = pointwise_scorer
    self.chunk_size = chunk_size

  def _scores_for_chunk(self, note, icd):
    padded_icd = [tens.to(self.device) for tens in collate_bow(icd)]
    padded_note = [tens.to(self.device) for tens in collate_bow(note)]
    with torch.no_grad():
      try:
        self.pointwise_scorer.eval()
        scores = self.pointwise_scorer(padded_note,
                                       padded_icd)
      finally:
        self.pointwise_scorer.train()
    return at_least_one_dim(scores)

  def __call__(self, note, icd, k=None):
    with torch.no_grad():
      k = k if k is not None else len(icd)
      ranks = []
      for note, icd in zip(note, icd):
        if self.chunk_size != -1:
          all_scores = []
          for from_idx, to_idx in zip(range(0,
                                            len(icd),
                                            self.chunk_size),
                                      range(self.chunk_size,
                                            len(icd) + self.chunk_size,
                                            self.chunk_size)):
            all_scores.append(self._scores_for_chunk(note,
                                                     icd[from_idx : to_idx]))
          scores = torch.cat(all_scores, 0)
        else:
          scores = self._scores_for_chunk(note, icd)
        topk_scores, topk_idxs = torch.topk(scores, k)
        sorted_scores, sort_idx = torch.sort(topk_scores, descending=True)
        ranks.append(topk_idxs[sort_idx])
      return torch.stack(ranks)
