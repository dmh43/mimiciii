from torch.utils.data import Dataset

class MimicDataset(Dataset):
  def __init__(self, notes_bow, icd_desc_bow, training_pairs):
    self.notes_bow, self.icd_desc_bow, self.training_pairs = notes_bow, icd_desc_bow, training_pairs

  def __len__(self):
    return len(self.training_pairs)

  def __getitem__(self, idx):
    note_id, pair = self.training_pairs(idx)
    return self.notes_bow[note_id], self.icd_desc_bow[pair[0]], self.icd_desc_bow[pair[1]]
