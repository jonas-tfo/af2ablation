#!/usr/bin/env python3

from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("poster") # Options: "paper", "notebook", "talk", "poster"

canonicals = ["custom", "alphamask"]

def _paths(method: str, target: str):
    if method in canonicals:
        base = Path(f"runs/{target}/{method}")
        return base / "tm_scores.csv", base / "plots"
    elif method == "afsample2": 
        base = Path(f"afsample_paper_models/generated_models/oc23/{method}/af_io_abl_15/{target}")
        return base / "final_df_tmalign.csv", base / "plots"
    else: 
        base = Path(f"afsample_paper_models/generated_models/oc23/{method}/{target}")
        return base / "final_df_tmalign.csv", base / "plots"


def tm_qmask(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    xcol = f"{xlabel}_tm"
    ycol = f"{ylabel}_tm"
    if method not in canonicals:
        xcol = "TM_open"
        ycol = "TM_close"
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=df["Query Mask %"].astype("category"),
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    plt.xlim((0.1, 1.02))
    plt.ylim((0.1, 1.02))
    plt.xlabel(f"TM {xlabel}")
    plt.ylabel(f"TM {ylabel}")
    plt.title("TM-score by Query Masking")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "tm_qmask.png")
    plt.close()


def tm_mmask(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    xcol = f"{xlabel}_tm"
    ycol = f"{ylabel}_tm"
    if method not in canonicals:
        xcol = "TM_open"
        ycol = "TM_close"
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=df["MSA Mask %"].astype("category"),
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    plt.xlim((0.1, 1.02))
    plt.ylim((0.1, 1.02))
    plt.xlabel(f"TM {xlabel}", fontdict={ "fontsize": 12 })
    plt.ylabel(f"TM {ylabel}")
    plt.title("TM-score by MSA Masking")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "tm_mmask.png")
    plt.close()


def tm_msub(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    xcol = f"{xlabel}_tm"
    ycol = f"{ylabel}_tm"
    if method not in canonicals:
        xcol = "TM_open"
        ycol = "TM_close"
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=df["MSA Samples"].astype("category"),
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    plt.xlim((0.1, 1.02))
    plt.ylim((0.1, 1.02))
    plt.xlabel(f"TM {xlabel}")
    plt.ylabel(f"TM {ylabel}")
    plt.title("TM-score by MSA Subsampling")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "tm_msub.png")
    plt.close()


def tm_qmask_mmask(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    xcol = f"{xlabel}_tm"
    ycol = f"{ylabel}_tm"
    if method not in canonicals:
        xcol = "TM_open"
        ycol = "TM_close"
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=df["Query Mask %"].astype("category"),
        style="MSA Mask %",
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    plt.xlim((0.1, 1.02))
    plt.ylim((0.1, 1.02))
    plt.xlabel(f"TM {xlabel}")
    plt.ylabel(f"TM {ylabel}")
    plt.title("TM-score by Query + MSA Masking")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "tm_qmask_mmask.png")
    plt.close()


def tm_qmask_msub(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    xcol = f"{xlabel}_tm"
    ycol = f"{ylabel}_tm"
    if method not in canonicals:
        xcol = "TM_open"
        ycol = "TM_close"
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=df["MSA Samples"].astype("category"),
        style="Query Mask %",
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    plt.xlim((0.1, 1.02))
    plt.ylim((0.1, 1.02))
    plt.xlabel(f"TM {xlabel}")
    plt.ylabel(f"TM {ylabel}")
    plt.title("TM-score by Query Masking + MSA Subsampling")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "tm_qmask_msub.png")
    plt.close()


def tm_best_marked(method: str, target: str, xlabel: str, ylabel: str, threshold: float = 0.9):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    xcol = f"{xlabel}_tm"
    ycol = f"{ylabel}_tm"
    if method not in canonicals:
        xcol = "TM_open"
        ycol = "TM_close"
    x = df[xcol]
    y = df[ycol]

    best = df[[xcol, ycol]].max(axis=1)
    mask = best >= threshold

    plt.figure(figsize=(7, 6))
    # full ensemble
    plt.scatter(x, y, s=15, color="lightgrey", alpha=0.6, linewidths=0, zorder=1)

    sc = plt.scatter(
        x[mask], y[mask],
        c=best[mask], cmap="viridis_r",
        s=70, edgecolor="black", linewidth=0.6, zorder=2
    )
    if mask.any():
        cbar = plt.colorbar(sc)
        cbar.set_label("TM-score of model to reference")

    lo = min(x.min(), y.min()) - 0.03
    plt.xlim(lo, 1.02)
    plt.ylim(lo, 1.02)
    plt.xlabel(f"TMscore (to {xlabel})")
    plt.ylabel(f"TMscore (to {ylabel})")
    n = int(mask.sum())
    plt.title(f"{target} : {n} match(es) with TM ≥ {threshold}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "tm_mapped.png")
    print("saved", target)
    plt.close()


def plot_all(method: str, target: str, xlabel: str, ylabel: str):
    if method in canonicals:
        tm_qmask(method, target, xlabel, ylabel)
        tm_mmask(method, target, xlabel, ylabel)
        tm_msub(method, target, xlabel, ylabel)
        tm_qmask_mmask(method, target, xlabel, ylabel)
        tm_qmask_msub(method, target, xlabel, ylabel)
    tm_best_marked(method, target, xlabel, ylabel)
