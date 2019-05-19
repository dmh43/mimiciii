import torch
import torch.nn as nn
from toolz import pipe

from .note_encoder import NoteEncoder
from .icd_encoder import ICDEncoder
from .utils import Identity

def _get_layer(from_size, to_size, dropout_keep_prob, activation=None, use_layer_norm=False, use_batch_norm=False):
  return [nn.Linear(from_size, to_size),
          nn.LayerNorm(to_size) if use_layer_norm else Identity(),
          nn.BatchNorm1d(to_size) if use_batch_norm else Identity(),
          nn.ReLU() if activation is None else activation,
          nn.Dropout(1 - dropout_keep_prob)]

class PointwiseScorer(nn.Module):
  def __init__(self,
               note_token_embeds,
               icd_token_embeds,
               model_params,
               train_params):
    super().__init__()
    self.use_layer_norm = train_params.use_layer_norm
    self.use_batch_norm = train_params.use_batch_norm
    self.icd_encoder = ICDEncoder(icd_token_embeds)
    self.note_encoder = NoteEncoder(note_token_embeds)
    concat_len = model_params.token_embed_len + model_params.token_embed_len
    self.layers = nn.ModuleList()
    from_size = concat_len + sum([model_params.token_embed_len
                                  for i in [model_params.append_hadamard, model_params.append_difference]
                                  if i])
    for to_size in model_params.hidden_layer_sizes:
      self.layers.extend(_get_layer(from_size,
                                    to_size,
                                    train_params.dropout_keep_prob,
                                    use_layer_norm=self.use_layer_norm,
                                    use_batch_norm=self.use_batch_norm))
      from_size = to_size
    self.layers.extend(_get_layer(from_size, 1, train_params.dropout_keep_prob, activation=Identity()))
    if not train_params.use_bce_loss:
      self.layers.append(nn.Tanh())
    self.append_difference = model_params.append_difference
    self.append_hadamard = model_params.append_hadamard


  def forward(self, note, icd):
    icd_embed = self.icd_encoder(icd)
    note_embed = self.note_encoder(note)
    hidden = torch.cat([icd_embed, note_embed], 1)
    if self.append_difference:
      hidden = torch.cat([hidden, torch.abs(icd_embed - note_embed)], 1)
    if self.append_hadamard:
      hidden = torch.cat([hidden, icd_embed * note_embed], 1)
    return pipe(hidden, *self.layers).reshape(-1)
