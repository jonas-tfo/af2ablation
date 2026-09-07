#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from afpert.evaluation.benchmark import summarize_run

THRESHOLD = 0.5
LOWER_LIMIT = 0.2
UPPER_LIMIT = 1.02

_PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#e377c2", "#ff7f0e", "#17becf", "#8c564b"]


def get_methods(target: str) -> list[str]:
    base = Path(f"runs/{target}")
    methods = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / "benchmark_summary.csv").exists() or (d / "tm_scores.csv").exists():
            methods.append(d.name)
    return methods


def _load(method: str, target: str) -> pd.DataFrame:
    csv = Path(f"runs/{target}/{method}/benchmark_summary.csv")
    if not csv.exists():
        summarize_run(target, method)
    return pd.read_csv(csv)


def panels(target: str, structure1: str, structure2: str, lim: tuple[float, float] = (LOWER_LIMIT, UPPER_LIMIT)):
    methods = get_methods(target)
    if not methods:
        structure1 = structure1.split("_")[0]
        structure2 = structure2.split("_")[0]
        methods = get_methods(target)
        if not methods:
            print(f"no methods with summaries found under runs/{target}")
            return

    xcol, ycol = f"{structure1}_tm_max", f"{structure2}_tm_max"

    fig, axes = plt.subplots(1, len(methods), figsize=(2.6 * len(methods) + 0.5, 3.0), squeeze=False, sharex=True, sharey=True)

    for i, (ax, method) in enumerate(zip(axes[0], methods)):
        df = _load(method, target)
        print(df.describe)
        if xcol not in df.columns or ycol not in df.columns:
            print(f"skipping {method}: {xcol!r}/{ycol!r} missing")
            print(df.describe())
            continue

        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter(df[xcol], df[ycol], c=color, s=70, edgecolor="black", linewidth=0.6, alpha=0.7, zorder=3)

        best = df.loc[(df[[xcol, ycol]].min(axis=1)).idxmax()]
        ax.scatter([best[xcol]], [best[ycol]], s=170, facecolor="none", edgecolor="black", linewidth=1.6, zorder=4)

        ax.axhline(THRESHOLD, ls="-.", lw=0.9, color="0.5", zorder=1)
        ax.axvline(THRESHOLD, ls="-.", lw=0.9, color="0.5", zorder=1)

        ax.set_title(method, fontsize=12)
        ax.set_xlabel("Best open")
        if i == 0:
            ax.set_ylabel("Best closed")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.text(0.04, 0.04, f"open {df[xcol].max():.2f}\nclosed {df[ycol].max():.2f}", transform=ax.transAxes, fontsize=8, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    fig.suptitle(f"Best fold reachability (TM-score)  [{target}]", y=1.02)
    fig.tight_layout()

    out = Path(f"runs/{target}/best_open_closed.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out