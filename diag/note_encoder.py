from toolz import pipe

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoteEncoder(nn.Module):
  def __init__(self, note_token_embeds):
    super().__init__()
    self.note_token_embeds = note_token_embeds
    self.weights = nn.Embedding(len(note_token_embeds.weight), 1)
    torch.nn.init.xavier_normal_(self.weights.weight.data)

  def forward(self, note):
    terms, cnts = note
    token_weights = self.weights(terms).squeeze() + torch.log(cnts.float())
    normalized_weights = F.softmax(token_weights, 1)
    note_tokens = self.note_token_embeds(terms)
    doc_vecs = torch.sum(normalized_weights.unsqueeze(2) * note_tokens, 1)
    encoded = doc_vecs
    return encoded
