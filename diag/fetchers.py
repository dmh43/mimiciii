from pathlib import Path
import pandas as pd

from .utils import to_lookup

def fetch_note_by_id(note_id, path='./data/notes/'):
  with open(Path(path).joinpath(str(note_id))) as fh:
    return fh.read()

def fetch_icd_desc_lookup(path='./data/D_ICD_DIAGNOSES.csv'):
  df = pd.read_csv(path)
  return to_lookup(df, 'ICD9_CODE', 'LONG_TITLE')
