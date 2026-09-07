#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# REF_RMSD = 3.534
REF_RMSD = 11.038


def _paths(method: str, target: str):
    base = Path(f"runs/{target}/{method}")
    return base / "rmsd_scores.csv", base / "plots"


def rmsd_qmask(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 5], [-1, 5], alpha=0.2)
    sns.scatterplot(
        data=df,
        x=f"{xlabel}_rmsd",
        y=f"{ylabel}_rmsd",
        hue=df["Query Mask %"].astype("category"),
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    sns.scatterplot(x=[0, REF_RMSD], y=[REF_RMSD, 0], marker="*", s=250)
    # plt.text(0, 3.7, f"{xlabel}")
    # plt.text(3.5, 0.2, f"{ylabel}")
    plt.xlim((-0.2, 15))
    plt.ylim((-0.2, 15))
    plt.xlabel(f"RMSD {xlabel}")
    plt.ylabel(f"RMSD {ylabel}")
    plt.title("RMSD by Query Masking")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "rmsd_qmask.png")
    plt.close()


def rmsd_mmask(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 5], [-1, 5], alpha=0.2)
    sns.scatterplot(
        data=df,
        x=f"{xlabel}_rmsd",
        y=f"{ylabel}_rmsd",
        hue=df["MSA Mask %"].astype("category"),
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    sns.scatterplot(x=[0, REF_RMSD], y=[REF_RMSD, 0], marker="*", s=250)
    # plt.text(0, 3.7, f"{xlabel}")
    # plt.text(3.5, 0.2, f"{ylabel}")
    plt.xlim((-0.2, 15))
    plt.ylim((-0.2, 15))
    plt.xlabel(f"RMSD {xlabel}")
    plt.ylabel(f"RMSD {ylabel}")
    plt.title("RMSD by MSA Masking")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "rmsd_mmask.png")
    plt.close()


def rmsd_msub(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 5], [-1, 5], alpha=0.2)
    sns.scatterplot(
        data=df,
        x=f"{xlabel}_rmsd",
        y=f"{ylabel}_rmsd",
        hue=df["MSA Samples"].astype("category"),
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    sns.scatterplot(x=[0, REF_RMSD], y=[REF_RMSD, 0], marker="*", s=250)
    # plt.text(0, 3.7, f"{xlabel}")
    # plt.text(3.5, 0.2, f"{ylabel}")
    plt.xlim((-0.2, 15))
    plt.ylim((-0.2, 15))
    plt.xlabel(f"RMSD {xlabel}")
    plt.ylabel(f"RMSD {ylabel}")
    plt.title("RMSD by MSA Subsampling")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "rmsd_msub.png")
    plt.close()


def rmsd_qmask_mmask(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 5], [-1, 5], alpha=0.2)
    sns.scatterplot(
        data=df,
        x=f"{xlabel}_rmsd",
        y=f"{ylabel}_rmsd",
        hue=df["Query Mask %"].astype("category"),
        style="MSA Mask %",
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    sns.scatterplot(x=[0, REF_RMSD], y=[REF_RMSD, 0], marker="*", s=250)
    # plt.text(0, 3.7, f"{xlabel}")
    # plt.text(3.5, 0.2, f"{ylabel}")
    plt.xlim((-0.2, 15))
    plt.ylim((-0.2, 15))
    plt.xlabel(f"RMSD {xlabel}")
    plt.ylabel(f"RMSD {ylabel}")
    plt.title("RMSD by Query + MSA Masking")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "rmsd_qmask_mmask.png")
    plt.close()


def rmsd_qmask_msub(method: str, target: str, xlabel: str, ylabel: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 5], [-1, 5], alpha=0.2)
    sns.scatterplot(
        data=df,
        x=f"{xlabel}_rmsd",
        y=f"{ylabel}_rmsd",
        hue=df["MSA Samples"].astype("category"),
        style="Query Mask %",
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    sns.scatterplot(x=[0, REF_RMSD], y=[REF_RMSD, 0], marker="*", s=250)
    # plt.text(0, 3.7, f"{xlabel}")
    # plt.text(3.5, 0.2, f"{ylabel}")
    plt.xlim((-0.2, 15))
    plt.ylim((-0.2, 15))
    plt.xlabel(f"RMSD {xlabel}")
    plt.ylabel(f"RMSD {ylabel}")
    plt.title("RMSD by Query Masking + MSA Subsampling")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "rmsd_qmask_msub.png")
    plt.close()


def plot(method: str, target: str, xlabel: str, ylabel: str, combination: str):
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 5], [-1, 5], alpha=0.2)
    sns.scatterplot(
        data=df,
        x=f"{xlabel}_rmsd",
        y=f"{ylabel}_rmsd",
        hue=df["MSA Samples"].astype("category"),
        style="Query Mask %",
        edgecolor="black",
        palette="tab10",
        alpha=0.3,
        s=100,
    )
    sns.scatterplot(x=[0, REF_RMSD], y=[REF_RMSD, 0], marker="*", s=250)
    # plt.text(0, 3.7, f"{xlabel}")
    # plt.text(3.5, 0.2, f"{ylabel}")
    plt.xlim((-0.2, 15))
    plt.ylim((-0.2, 15))
    plt.xlabel(f"RMSD {xlabel}")
    plt.ylabel(f"RMSD {ylabel}")
    plt.title("RMSD by Query Masking + MSA Subsampling")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "rmsd_qmask_msub.png")
    plt.close()


def plot_all(method: str, target: str, xlabel: str, ylabel: str):
    rmsd_qmask(method, target, xlabel, ylabel)
    rmsd_mmask(method, target, xlabel, ylabel)
    rmsd_msub(method, target, xlabel, ylabel)
    rmsd_qmask_mmask(method, target, xlabel, ylabel)
    rmsd_qmask_msub(method, target, xlabel, ylabel)
