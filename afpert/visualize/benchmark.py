#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from afpert.evaluation.benchmark import summarize_run

ROW, COL, FACET = "MSA Mask %", "Query Mask %", "MSA Samples"


def _paths(method: str, target: str):
    base = Path(f"runs/{target}/{method}")
    return base / "benchmark_summary.csv", base / "plots"


def _load(method: str, target: str) -> pd.DataFrame:
    csv, _ = _paths(method, target)
    if not csv.exists():
        summarize_run(target, method)
    return pd.read_csv(csv)


def _states(df: pd.DataFrame):
    return [c[len("frac_"):] for c in df.columns if c.startswith("frac_")]


def heatmap_grid(method: str, target: str, value: str, *, vmin=None, vmax=None, cmap="viridis", fmt=".2f"):
    df = _load(method, target)
    if value not in df.columns:
        print(f"skipping {value!r}: not in benchmark_summary.csv")
        return

    depths = sorted(df[FACET].unique())
    # share colour scale across facets 
    if vmin is None:
        vmin = df[value].min()
    if vmax is None:
        vmax = df[value].max()

    fig, axes = plt.subplots(1, len(depths), figsize=(5 * len(depths), 7), squeeze=False)
    for ax, depth in zip(axes[0], depths):
        grid = (
            df[df[FACET] == depth]
            .pivot_table(index=ROW, columns=COL, values=value, aggfunc="mean")
            .sort_index(ascending=False)
        )
        sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=fmt, cbar=False, linewidths=0.5, square=True)
        ax.set_title(f"{FACET} = {depth}")

    fig.suptitle(f"{value}   [{target} / {method}]")
    fig.tight_layout()

    _, plots_dir = _paths(method, target)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"benchmark_{value}.png", dpi=150)
    plt.close(fig)


def plot_all(method: str, target: str):
    df = _load(method, target)
    states = _states(df)

    heatmap_grid(method, target, "fold_entropy", vmin=0.0, vmax=1.0)
    for s in states:
        heatmap_grid(method, target, f"frac_{s}", vmin=0.0, vmax=1.0)

    for s in states:
        heatmap_grid(method, target, f"{s}_tm_max", cmap="viridis")
        heatmap_grid(method, target, f"{s}_rmsd_min", cmap="viridis_r")
