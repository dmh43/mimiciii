import torch
import torch.nn as nn

from .pointwise_scorer import PointwiseScorer

class PairwiseScorer(nn.Module):
  def __init__(self,
               note_token_embeds,
               icd_token_embeds,
               model_params,
               train_params):
    super().__init__()
    self.pointwise_scorer = PointwiseScorer(note_token_embeds,
                                            icd_token_embeds,
                                            model_params,
                                            train_params)

  def forward(self, note, icd_1, icd_2):
    score_1 = self.pointwise_scorer(note, icd_1)
    score_2 = self.pointwise_scorer(note, icd_2)
    return score_1 - score_2
