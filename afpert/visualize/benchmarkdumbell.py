#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from afpert.evaluation.benchmark import summarize_run

FACET = "MSA Samples"
ROW_KEYS = ["MSA Mask %", "Query Mask %"]


def _paths(method: str, target: str):
    base = Path(f"runs/{target}/{method}")
    return base / "benchmark_summary.csv", base / "plots"


def _load(method: str, target: str) -> pd.DataFrame:
    csv, _ = _paths(method, target)
    if not csv.exists():
        summarize_run(target, method)
    return pd.read_csv(csv)


def _states(df: pd.DataFrame):
    """Referenzkonformation namen"""
    return [c[len("frac_"):] for c in df.columns if c.startswith("frac_")]


def _row_label(row: pd.Series) -> str:
    return f"M{row['MSA Mask %']:g}% / Q{row['Query Mask %']:g}%"


def dumbbell(method: str, target: str, metric: str, xlabel: str, ylabel: str):
    df = _load(method, target)
    states = _states(df)
    if len(states) < 2:
        return

    a, b = states[0], states[1]
    col_a, col_b = f"{a}_{metric}", f"{b}_{metric}"
    if col_a not in df.columns or col_b not in df.columns:
        return

    higher_better = metric.endswith("max")
    xlabel = "TM-score" if "tm" in metric else "RMSD (Angstrom)"

    depths = sorted(df[FACET].unique())
    fig, axes = plt.subplots(
        1, len(depths), figsize=(5.5 * len(depths), 7), squeeze=False, sharex=True
    )

    ca, cb = "#1f77b4", "#d62728"

    for ax, depth in zip(axes[0], depths):
        sub = df[df[FACET] == depth].copy()
        sep = sub[col_a] - sub[col_b]
        sub["_sep"] = sep if higher_better else -sep
        sub = sub.sort_values("_sep").reset_index(drop=True)
        sub = sub.sort_values(by=['Query Mask %', 'MSA Mask %'], ascending=False)

        y = range(len(sub))
        ax.hlines(y, sub[col_a], sub[col_b], color="0.7", lw=2, zorder=1)
        ax.scatter(sub[col_a], y, color=ca, s=70, zorder=2, label=a)
        ax.scatter(sub[col_b], y, color=cb, s=70, zorder=2, label=b)

        ax.set_yticks(list(y))
        ax.set_yticklabels([_row_label(r) for _, r in sub.iterrows()], fontsize=8)
        ax.set_title(f"{FACET} = {depth}")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
        ax.margins(y=0.02)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], title="reference fold", loc="upper right", frameon=True)
    fig.suptitle(f"Fold reachability per setting   [{target} / {method}]")
    fig.tight_layout()

    _, plots_dir = _paths(method, target)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"overview_dumbbell_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_all(method: str, target: str, xlabel: str, ylabel: str):
    dumbbell(method, target, "tm_max", xlabel, ylabel)
    dumbbell(method, target, "rmsd_min", xlabel, ylabel)
