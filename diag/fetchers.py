from pathlib import Path
import pandas as pd

from .utils import to_lookup, get_token_cnts

def fetch_note_by_id(note_id, path='./data/notes/'):
  with open(Path(path).joinpath(f'note_{note_id}')) as fh:
    return fh.read()

def fetch_icd_desc_lookup(path='./data/D_ICD_DIAGNOSES.csv'):
  df = pd.read_csv(path)
  return to_lookup(df, 'ICD9_CODE', 'LONG_TITLE')

def get_token_set(tokenizer, icd_desc_lookup_by_label, opts):
  min_num_occurances = opts['min_num_occurances']
  notes_df = pd.read_csv('data/notes.csv')
  notes_df.set_index('note_id', inplace=True)
  diagnoses_df = pd.read_csv('data/diagnoses.csv')
  notes_df = notes_df.loc[diagnoses_df.note_id]
  notes_df['note_id'] = notes_df.index
  notes_lookup = to_lookup(notes_df, 'note_id', 'text')
  token_cnts = get_token_cnts(tokenizer, list(notes_lookup.values()))
  icd_token_cnts = get_token_cnts(tokenizer, list(icd_desc_lookup_by_label.values()))
  token_set = {token
               for token, cnt in token_cnts.items()
               if cnt + icd_token_cnts[token] >= min_num_occurances}
  token_set.update(icd_token_cnts.keys())
  return token_set
