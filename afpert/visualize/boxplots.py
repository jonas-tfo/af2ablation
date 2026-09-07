#!/usr/bin/env python3

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SUMMARY_GLOB = "runs/**/benchmark_summary.csv"
# OPEN_CLOSED: dict[str, tuple[str, str]] = {
#     "7DSQ_2": ("7DSQ", "6IRS"),
# }

SHARED_TARGETS_ONLY = True

DOWNLOADED_GLOB = "afsample_paper_models/generated_models/**/final_df_tmalign.csv"
TARGETS: list[str] = [
    "A0A075Q0W3", 
    "A0A1J6PWI8",
    "A0QTT2",
    "A0R629",
    "A2RJ53",
    "A6UVT1",
    "B3EYN2",
    "B7IE18",
    "C7C425",
    "J9UN47",
    "O34926",
    "O76728",
    "P00558",
    "P00918",
    "P0A4G2",
    "P0CG48",
    "P14902",
    "P15291",
    "P18031",
    "P18965",
    "P21589",
    "P29350",
    "P31133",
    "P33284",
    "P40131",
    "P48635",
    "P61316",
    "P62495",
    "P71447",
    "P76045",
    "P9WNX1",
    "Q18A65",
    "Q53W80",
    "Q5F9M1",
    "Q72HW2",
    "Q7DAU8",
    "Q82GL5",
    "Q8A5V9",
    "Q9ERE7",
    "Q9SS90",
    "Q9U6Y3",
    "Q9UBV7",
    "Q9X6R4",
    "Q9X9P9",
    "Q9Z4N6",
    "PF0708",
]

# AGG = "max"
AGG = "max"

OUT_PATH = "runs/method_boxplots.png"

DISPLAY_LABELS: dict[str, str] = {
    "custom": "Custom (ours)",
    "afsample": "AFsample [1]",
    "SPEACH_AF": "SPEACH_AF [2]",
    "msasubsampling": "MSA Subsampling [3]",
    "afsample2": "AFSample2 [4]",
    "AFSample2": "AFSample2 [4]",
    "afvanilla": "Vanilla AF2 [5]",
}


def display_label(method: str) -> str:
    """Map a real method name (possibly ``paper_``-prefixed) to its display label."""
    return DISPLAY_LABELS.get(method) or DISPLAY_LABELS.get(
        method.removeprefix("paper_"), method
    )


def _tm_max_col(columns, struct: str) -> str | None:
    want = f"{struct}_tm_max"
    lower = {c.lower(): c for c in columns}
    return lower.get(want.lower())


def get_downloaded_open_closed(pattern: str = DOWNLOADED_GLOB, targets: list[str] = TARGETS) -> dict[str, tuple[str, str]]:
    """get open_id, closed_id from pdbid_o/pdbid_c"""
    tset = set(targets)
    oc: dict[str, tuple[str, str]] = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        target = next((p for p in Path(f).parts if p in tset), None)
        if target is None or target in oc:
            continue
        head = pd.read_csv(f, nrows=1)
        oc[target] = (str(head["pdbid_o"].iloc[0]), str(head["pdbid_c"].iloc[0]))
    return oc


def load_summary_evals(open_closed: dict[str, tuple[str, str]], pattern: str = SUMMARY_GLOB, agg: str = AGG) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(pattern, recursive=True)): # all benchmark_summary.csv
        method = Path(f).parent.name
        df = pd.read_csv(f)
        for tname, sub in df.groupby("target"):
            oc = open_closed.get(str(tname))
            if oc is None:
                continue
            for state, struct in zip(("open", "closed"), oc):
                col = _tm_max_col(sub.columns, struct)
                if col is None or sub[col].dropna().empty:
                    print(f"skipping {method}/{tname}/{state}: column {struct}_tm_max missing")
                    continue
                val = sub[col].max() if agg == "max" else sub[col].mean()
                if method != "afsample2":
                    rows.append({"method": method, "target": str(tname), "state": state, "tm": float(val)})
    return pd.DataFrame(rows)


def load_downloaded_evals(pattern: str = DOWNLOADED_GLOB, agg: str = AGG, targets: list[str] = TARGETS) -> pd.DataFrame:
    tset = set(targets)
    rows = []
    for f in sorted(glob.glob(pattern, recursive=True)):  # all final_df.csv
        parts = Path(f).parts
        target = next((p for p in parts if p in tset), None)
        if target is None:
            print(f"skipping {f}: no known target in path")
            continue
        try:  # method = dir right under the dataset folder
            method = parts[parts.index("generated_models") + 2]
        except (ValueError, IndexError):
            method = Path(f).parent.parent.name
        method = f"paper_{method}"  # paper and own implementation sep for now maybe good for comparison

        df = pd.read_csv(f)
        for state, col in (("open", "TM_open"), ("closed", "TM_close")):
            if col not in df.columns or df[col].dropna().empty:
                print(f"skipping {method}/{target}/{state}: column {col!r} missing")
                continue
            val = df[col].max() if agg == "max" else df[col].mean()
            rows.append({"method": method, "target": target, "state": state, "tm": float(val)})

    out = pd.DataFrame(rows)
    if not out.empty:  # collapse multiple files per (method, target) for afsample2
        out = out.groupby(["method", "target", "state"], as_index=False)["tm"].agg(agg)
    return out


def filter_shared_targets(long: pd.DataFrame) -> pd.DataFrame:
    n_methods = long["method"].nunique()
    counts = long.groupby("target")["method"].nunique()
    keep = counts[counts == n_methods].index
    dropped = sorted(set(long["target"]) - set(keep))
    if dropped:
        print(f"dropping {len(dropped)} target(s) not shared by all {n_methods} methods: {dropped}")
    return long[long["target"].isin(keep)]


def merge(agg: str = AGG) -> pd.DataFrame:
    open_closed = get_downloaded_open_closed(DOWNLOADED_GLOB, TARGETS)
    long = pd.concat(
        [load_summary_evals(open_closed, SUMMARY_GLOB, agg), load_downloaded_evals(DOWNLOADED_GLOB, agg)],
        ignore_index=True,
    )
    if SHARED_TARGETS_ONLY and not long.empty:
        long = filter_shared_targets(long)
    return long


def boxplot(ax, long_df: pd.DataFrame, state: str, order: list[str], ylabel: str) -> None:
    data = long_df[long_df["state"] == state]
    # print(data.describe)
    sns.set_context("talk")
    sns.boxplot(
        data=data, x="method", y="tm", order=order, ax=ax,
        showfliers=False, width=0.6,
        boxprops=dict(facecolor="none", edgecolor="0.4"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"),
        # whis=(0,100)
    )
    sns.stripplot(
        data=data, x="method", y="tm", order=order, ax=ax,
        hue="method", legend=False,
        jitter=0.05, size=5, alpha=0.7, edgecolor="black", linewidth=0.3,
    )

    ax.axhline(1.0, color="black", lw=1.0, zorder=1)
    ax.set_ylim(0.59, 1.01)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Best models ({state.capitalize()} state)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [display_label(m) for m in order],
        rotation=20, ha="right", rotation_mode="anchor",
    )
    ax.grid(axis="x", ls="-.", lw=0.8, color="0.85")


def plot(agg: str = AGG, out_path: str = OUT_PATH) -> Path | None:
    long_df = merge(agg)
    if long_df.empty:
        print("no data loaded; nothing to plot")
        return None

    order = [
        "custom",
        "paper_afsample",
        "paper_SPEACH_AF",
        "paper_msasubsampling",
        "paper_afsample2",
        "paper_afvanilla",
    ]
    present = set(long_df["method"].unique())
    order = [m for m in order if m in present]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    boxplot(axes[0], long_df, "open", order, "TM-score to open")
    boxplot(axes[1], long_df, "closed", order, "TM-score to close")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    plot()
