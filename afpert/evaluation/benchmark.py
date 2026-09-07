#!/usr/bin/env python3

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PARAM_COLS = ["MSA Samples", "Query Mask %", "MSA Mask %"]

def _metric_cols(df: pd.DataFrame, suffix: str) -> List[str]:
    return [c for c in df.columns if c.endswith(suffix)]


def _agg_metric(g: pd.DataFrame, cols: List[str], stats=("mean", "std", "min", "max")) -> dict:
    out = {}
    for c in cols:
        for s in stats:
            out[f"{c}_{s}"] = getattr(g[c], s)()
    return out


def summarize_run(target: str, method: str, runs_root: str | Path = "runs") -> Optional[pd.DataFrame]:
    """
    Build per-combination benchmark stats for one target/method and write
    TM and RMSD aggregated independently and joined on the combination key,
    """
    base = Path(runs_root) / target / method
    tm_path = base / "tm_scores.csv"
    rmsd_path = base / "rmsd_scores.csv"
    if not tm_path.exists():
        return None

    tm = pd.read_csv(tm_path)
    tm_cols = _metric_cols(tm, "_tm")
    states = [c[:-3] for c in tm_cols]  # eg "7DSQ", "6IRS"
    
    rows = []
    for key, g in tm.groupby(PARAM_COLS, sort=True):
        row = dict(zip(PARAM_COLS, key))
        row["n_pred"] = len(g)
        row.update(_agg_metric(g, tm_cols))
        if "mean_plddt" in g:
            row["mean_plddt"] = g["mean_plddt"].mean()

        # which reference conformation each prediction is closest to based on tm
        if len(tm_cols) >= 2:
            assigned = g[tm_cols].idxmax(axis=1).str[:-3]
            fracs = assigned.value_counts(normalize=True)
            for s in states:
                row[f"frac_{s}"] = float(fracs.get(s, 0.0))
            # 1.0 = perfectly split, 0.0 = entirely one fold
            p = np.array([fracs.get(s, 0.0) for s in states])
            p = p[p > 0]
            row["fold_entropy"] = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)

    # join RMSD stats on the combination key
    if rmsd_path.exists():
        rmsd = pd.read_csv(rmsd_path)
        rmsd_cols = _metric_cols(rmsd, "_rmsd")
        rstats = (
            rmsd.groupby(PARAM_COLS, sort=True)
            .apply(lambda g: pd.Series(_agg_metric(g, rmsd_cols)), include_groups=False)
            .reset_index()
        )
        summary = summary.merge(rstats, on=PARAM_COLS, how="left")

    summary.insert(0, "method", method)
    summary.insert(0, "target", target)
    summary = summary.sort_values(PARAM_COLS).reset_index(drop=True)
    summary.to_csv(base / "benchmark_summary.csv", index=False)
    return summary


def benchmark_all(
    runs_root: str | Path = "runs",
    methods: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    runs_root = Path(runs_root)
    frames = []
    for target_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for method_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            if methods is not None and method not in methods:
                continue
            s = summarize_run(target_dir.name, method, runs_root)
            if s is not None:
                frames.append(s)
                print(f"  {target_dir.name}/{method}: {len(s)} combinations")

    if not frames:
        raise SystemExit(f"no score CSVs found under {runs_root}/")

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(runs_root / "benchmark.csv", index=False)
    print(f"wrote {len(master)} combination rows to {runs_root / 'benchmark.csv'}")
    return master
