#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# perturbation-combo key, kept in sync with evaluation.rmsf / evaluation.benchmark.
# defined locally so plotting doesn't import biotite via the evaluation layer.
PARAM_COLS = ["MSA Samples", "Query Mask %", "MSA Mask %"]


def _paths(method: str, target: str):
    base = Path(f"runs/{target}/{method}")
    return base / "rmsf_scores.csv", base / "plots"


def _combo_label(row) -> str:
    """Compact perturbation-combo tag, e.g. ``S256/Q15/M0``."""
    return f"S{int(row['MSA Samples'])}/Q{int(row['Query Mask %'])}/M{int(row['MSA Mask %'])}"


def _corr(df: pd.DataFrame):
    """Pearson/Spearman of RMSF vs experimental displacement over matched residues."""
    paired = df.dropna(subset=["rmsf", "displacement"])
    if len(paired) >= 2:
        r, rp = pearsonr(paired["rmsf"], paired["displacement"])
        rho, rhop = spearmanr(paired["rmsf"], paired["displacement"])
    else:
        r = rp = rho = rhop = float("nan")
    return paired, r, rp, rho, rhop


def _ci_band(df: pd.DataFrame):
    """Lower/upper RMSF band, falling back to the point estimate if no CI columns."""
    lo = df["rmsf_lo"] if "rmsf_lo" in df else df["rmsf"]
    hi = df["rmsf_hi"] if "rmsf_hi" in df else df["rmsf"]
    return lo, hi


def plot(method: str, target: str):
    """Per-residue RMSF (with bootstrap CI) next to experimental Ca displacement.

    Left: RMSF along the sequence with its bootstrap CI as a shaded band, and the
    experimental Ca displacement on a twin axis so co-varying regions line up.
    Right: scatter of the two with Pearson/Spearman annotated.
    """
    csv, plots_dir = _paths(method, target)
    df = pd.read_csv(csv)
    paired, r, rp, rho, rhop = _corr(df)
    lo, hi = _ci_band(df)

    sns.set_theme(style="whitegrid")
    fig, (ax_line, ax_sc) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- per-residue traces ---
    c_rmsf, c_disp = "tab:blue", "tab:red"
    ax_line.fill_between(
        df["residue"],
        lo,
        hi,
        color=c_rmsf,
        alpha=0.25,
        lw=0,
        label="RMSF 95% CI",
    )
    ax_line.plot(df["residue"], df["rmsf"], color=c_rmsf, lw=1.5, label="RMSF")
    ax_line.set_xlabel("Residue")
    ax_line.set_ylabel("RMSF (Å)", color=c_rmsf)
    ax_line.tick_params(axis="y", labelcolor=c_rmsf)

    ax_disp = ax_line.twinx()
    ax_disp.grid(False)
    ax_disp.plot(
        df["residue"],
        df["displacement"],
        color=c_disp,
        lw=1.5,
        alpha=0.8,
        label="C-alpha displacement",
    )
    ax_disp.set_ylabel("Experimental C-alpha displacement (Å)", color=c_disp)
    ax_disp.tick_params(axis="y", labelcolor=c_disp)
    n_pred = int(df["n_predictions"].iloc[0]) if "n_predictions" in df else len(df)
    ax_line.set_title(
        f"Per-residue RMSF vs. experimental displacement : {target} / {method} "
        f"(n={n_pred} predictions)"
    )

    # --- correlation scatter ---
    sns.scatterplot(
        data=paired,
        x="displacement",
        y="rmsf",
        hue="residue",
        palette="viridis",
        legend=False,
        edgecolor="black",
        s=45,
        alpha=0.8,
        ax=ax_sc,
    )
    if len(paired) >= 2:
        b, a = np.polyfit(paired["displacement"], paired["rmsf"], 1)
        xs = np.array([paired["displacement"].min(), paired["displacement"].max()])
        ax_sc.plot(xs, a + b * xs, color="black", ls="--", lw=1, alpha=0.6)
    ax_sc.set_xlabel("Experimental C-alpha displacement (Å)")
    ax_sc.set_ylabel("RMSF (Å)")
    ax_sc.set_title(
        f"Pearson r = {r:.2f} (p={rp:.1e})\nSpearman rho = {rho:.2f} (p={rhop:.1e})"
    )

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "rmsf_displacement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (Pearson r={r:.3f}, Spearman rho={rho:.3f}, n={len(paired)})")


def _discover_methods(target: str) -> list[str]:
    base = Path(f"runs/{target}")
    if not base.exists():
        return []
    return [
        d.name
        for d in sorted(p for p in base.iterdir() if p.is_dir())
        if (d / "rmsf_scores.csv").exists()
    ]


def compare_per_residue_rmsf(target: str, methods: list[str] | None = None):
    """Benchmarking view: overlay each perturbation method's RMSF for one target.

    Every method's per-residue RMSF curve (with its bootstrap CI band) is drawn on
    a shared axis against the experimental Ca displacement (grey reference). A method
    whose RMSF tracks the displacement is exploring the right conformational
    degrees of freedom, so the per-method RMSF↔displacement correlations are
    summarised in the legend — that's the quantity that ranks the methods.
    """
    methods = methods or _discover_methods(target)
    if not methods:
        print(f"skipping rmsf compare: no rmsf_scores.csv under runs/{target}")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=len(methods))

    disp_ref = None
    for color, method in zip(palette, methods):
        csv, _ = _paths(method, target)
        df = pd.read_csv(csv)
        paired, r, _, rho, _ = _corr(df)
        lo, hi = _ci_band(df)
        ax.fill_between(df["residue"], lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(
            df["residue"],
            df["rmsf"],
            color=color,
            lw=1.6,
            label=f"{method}  (r={r:.2f}, \u03c1={rho:.2f})",
        )
        if disp_ref is None and df["displacement"].notna().any():
            disp_ref = df[["residue", "displacement"]].copy()

    if disp_ref is not None:
        ax_disp = ax.twinx()
        ax_disp.grid(False)
        ax_disp.plot(
            disp_ref["residue"],
            disp_ref["displacement"],
            color="dimgray",
            lw=1.4,
            ls="--",
            alpha=0.7,
        )
        ax_disp.set_ylabel("Experimental C-alpha displacement (Å)", color="dimgray")
        ax_disp.tick_params(axis="y", labelcolor="dimgray")

    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(
        f"Per-residue RMSF by perturbation method : {target}\n"
        "shaded = bootstrap 95% CI \n dashed grey = experimental C-alpha displacement \n ",
        fontdict={"fontsize": 14}
    )
    ax.legend(title="Method (Pearson r, Spearman \u03c1)", loc="best", framealpha=0.9)

    fig.tight_layout()
    out = Path(f"runs/{target}/rmsf_compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({len(methods)} methods: {', '.join(methods)})")


def compare_combos(method: str, target: str):
    """Per-target benchmarking: overlay each perturbation combo's RMSF curve.

    One curve (with its bootstrap CI band) per ``(MSA Samples, Query Mask %,
    MSA Mask %)`` combination that has predictions for this target/method, drawn
    against the experimental Cα displacement. The legend reports each combo's
    RMSF↔displacement correlation, so you can read off which perturbation setting
    best recovers the experimental motion for this target.
    """
    base = Path(f"runs/{target}/{method}")
    csv = base / "rmsf_by_combo.csv"
    if not csv.exists():
        print(
            f"skipping rmsf combo compare: {csv} not found (run build_rmsf_db with per_combo=True)"
        )
        return
    df = pd.read_csv(csv)
    combos = list(df.groupby(PARAM_COLS, sort=True))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=max(len(combos), 1))

    disp_ref = None
    for color, (_, g) in zip(palette, combos):
        g = g.sort_values("residue")
        _, r, _, rho, _ = _corr(g)
        lo, hi = _ci_band(g)
        ax.fill_between(g["residue"], lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(
            g["residue"],
            g["rmsf"],
            color=color,
            lw=1.5,
            label=f"{_combo_label(g.iloc[0])}  (r={r:.2f}, rho={rho:.2f})",
        )
        if disp_ref is None and g["displacement"].notna().any():
            disp_ref = g[["residue", "displacement"]].copy()

    if disp_ref is not None:
        ax_disp = ax.twinx()
        ax_disp.grid(False)
        ax_disp.plot(
            disp_ref["residue"],
            disp_ref["displacement"],
            color="dimgray",
            lw=1.4,
            ls="--",
            alpha=0.7,
        )
        ax_disp.set_ylabel(
            "Experimental C-alpha displacement (Angstrom)", color="dimgray"
        )
        ax_disp.tick_params(axis="y", labelcolor="dimgray")

    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(
        f"Per-residue RMSF by perturbation combo : {target} / {method}\n"
        "shaded = bootstrap 95% CI; dashed grey = experimental C-alpha displacement",
        fontdict={"fontsize":18}
    )
    ax.legend(
        title="S=Samples Q=Query% M=MSA% (r, alpha)",
        loc="best",
        fontsize=14,
        framealpha=0.9,
    )

    fig.tight_layout()
    (base / "plots").mkdir(parents=True, exist_ok=True)
    out = base / "plots" / "rmsf_by_combo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({len(combos)} combos)")


def _discover_targets(runs_root: str, methods: list[str] | None):
    """(target, method) pairs that have an rmsf_by_combo.csv under ``runs_root``."""
    root = Path(runs_root)
    pairs = []
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            if methods is not None and mdir.name not in methods:
                continue
            if (mdir / "rmsf_by_combo.csv").exists():
                pairs.append((tdir.name, mdir.name))
    return pairs


def aggregate(
    targets: list[str] | None = None,
    methods: list[str] | None = None,
    runs_root: str = "runs",
    metric: str = "spearman",
):
    pairs = _discover_targets(runs_root, methods)
    if targets is not None:
        pairs = [(t, m) for (t, m) in pairs if t in targets]
    if not pairs:
        print(f"skipping rmsf aggregate: no rmsf_by_combo.csv under {runs_root}/")
        return

    corr_fn = spearmanr if metric == "spearman" else pearsonr
    rows = []
    for target, method in pairs:
        df = pd.read_csv(Path(runs_root) / target / method / "rmsf_by_combo.csv")
        for _, g in df.groupby(PARAM_COLS, sort=True):
            paired = g.dropna(subset=["rmsf", "displacement"])
            if len(paired) < 2 or paired["rmsf"].nunique() < 2:
                continue
            stat = corr_fn(paired["rmsf"], paired["displacement"])[0]
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "combo": _combo_label(g.iloc[0]),
                    "corr": float(stat),
                }
            )

    if not rows:
        print("skipping rmsf aggregate: no combos with enough paired residues")
        return
    agg = pd.DataFrame(rows)
    # order combos by median correlation so the best perturbation reads left-to-right
    order = (
        agg.groupby("combo")["corr"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(order)), 6))
    multi_method = agg["method"].nunique() > 1
    hue = "method" if multi_method else None
    print(agg.describe)
    sns.boxplot(
        data=agg,
        x="combo",
        # y="corr",
        y="target",
        order=order,
        hue=hue,
        showfliers=False,
        width=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=agg,
        x="combo",
        # y="corr",
        y="target",
        order=order,
        hue=hue,
        dodge=multi_method,
        jitter=0.2,
        size=4,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="grey", lw=1, ls=":")
    ax.set_xlabel("Perturbation combo  (S=MSA Samples, Q=Query%, M=MSA%)")
    ax.set_ylabel(f"RMSF displacement correlation ({metric}, one point per target)")
    ax.set_title(
        f"Perturbation effect across {agg['target'].nunique()} targets "
        "how well each combo's RMSF tracks the experimental C-alpha displacement"
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out = Path(runs_root) / "rmsf_aggregate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    agg.to_csv(Path(runs_root) / "rmsf_aggregate.csv", index=False)
    print(
        f"wrote {out} and rmsf_aggregate.csv "
        f"({len(agg)} (target,combo) points, {agg['target'].nunique()} targets)"
    )

def _discover_target_method_pairs(runs_root: str, methods: list[str] | None):
    """(target, method) pairs with a pooled ``rmsf_scores.csv`` under ``runs_root``."""
    root = Path(runs_root)
    pairs = []
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            if methods is not None and mdir.name not in methods:
                continue
            if (mdir / "rmsf_scores.csv").exists():
                pairs.append((tdir.name, mdir.name))
    return pairs
 
 
def summary_over_targets(
    targets: list[str] | None = None,
    methods: list[str] | None = None,
    runs_root: str = "runs",
    metric: str = "spearman",
 ):
    pairs = _discover_target_method_pairs(runs_root, methods)
    if targets is not None:
        pairs = [(t, m) for (t, m) in pairs if t in targets]
    if not pairs:
        print(f"skipping rmsf summary: no rmsf_scores.csv under {runs_root}/")
        return
 
    corr_fn = spearmanr if metric == "spearman" else pearsonr
 
    def _corr_of(df: pd.DataFrame) -> float:
        paired = df.dropna(subset=["rmsf", "displacement"])
        if len(paired) < 2 or paired["rmsf"].nunique() < 2:
            return float("nan")
        return float(corr_fn(paired["rmsf"], paired["displacement"])[0])
 
    bar_rows = []   # pooled headline per (target, method)
    combo_rows = []  # per-combo points where available
    for target, method in pairs:
        base = Path(runs_root) / target / method
        pooled = _corr_of(pd.read_csv(base / "rmsf_scores.csv"))
        bar_rows.append({"target": target, "method": method, "corr": pooled})
 
        combo_csv = base / "rmsf_by_combo.csv"
        if combo_csv.exists():
            cdf = pd.read_csv(combo_csv)
            for _, g in cdf.groupby(PARAM_COLS, sort=True):
                c = _corr_of(g)
                if c == c:  # not NaN
                    combo_rows.append({"target": target, "method": method, "corr": c})
 
    bars = pd.DataFrame(bar_rows)
    combos = pd.DataFrame(combo_rows)
 
    order = (
        bars.groupby("target")["corr"].max().sort_values(ascending=False).index.tolist()
    )
    multi_method = bars["method"].nunique() > 1
    hue = "method" if multi_method else None
 
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(order)), 6))
    sns.barplot(
        data=bars,
        x="target",
        y="corr",
        order=order,
        hue=hue,
        edgecolor="black",
        alpha=0.85,
        ax=ax,
    )
    if not combos.empty:
        sns.stripplot(
            data=combos,
            x="target",
            y="corr",
            order=order,
            hue=hue,
            dodge=multi_method,
            jitter=0.2,
            size=4,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.3,
            ax=ax,
            legend=False,
        )
    ax.axhline(0, color="grey", lw=1, ls=":")
    ax.set_xlabel("Target")
    ax.set_ylabel(f"RMSF↔displacement correlation ({metric})")
    ax.set_title(
        f"RMSF performance across {bars['target'].nunique()} targets — "
        "how well each target's induced flexibility tracks the experimental "
        "Cα displacement\n(bar = pooled over all predictions "
        "points = per perturbation combo where available)"
    )
    ax.tick_params(axis="x", rotation=45)
    if multi_method:
        ax.legend(title="Method", loc="best", framealpha=0.9)
    fig.tight_layout()
 
    out = Path(runs_root) / "rmsf_summary_by_target.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    bars.to_csv(Path(runs_root) / "rmsf_summary_by_target.csv", index=False)
    print(
        f"wrote {out} and rmsf_summary_by_target.csv "
        f"({len(bars)} (target,method) bars, {bars['target'].nunique()} targets)"
    )
 
 
def plot_all(method: str, target: str):
    plot(method, target)
    compare_combos(method, target)