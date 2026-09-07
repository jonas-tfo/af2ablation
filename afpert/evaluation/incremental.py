"""

on later run skip pdbs already present in the csv and append only the new ones
"""

from pathlib import Path

import pandas as pd


def load_done(out_csv: str, incremental: bool):
    """
    Return previous_df_or_None, set_of_scored_pdb_paths)
    """
    if not incremental:
        return None, set()
    p = Path(out_csv)
    if not p.exists():
        return None, set()
    prev = pd.read_csv(p)
    # drop the unnamed index column older csvs were written with
    prev = prev.loc[:, ~prev.columns.str.startswith("Unnamed")]
    if "path" not in prev.columns:
        return None, set()
    return prev, set(prev["path"].astype(str))


def merge_and_write(prev_df, new_df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    if prev_df is not None and len(prev_df):
        combined = pd.concat([prev_df, new_df], ignore_index=True)
    else:
        combined = new_df
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined
