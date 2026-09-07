#!/usr/bin/env python3
"""Scrollable overview of every per-residue RMSF plot.

Discovers all ``runs/<target>/<method>/rmsf_scores.csv`` files and renders the
per-residue RMSF-vs-experimental-displacement plot for each one inline, so they
can be scrolled through in a single notebook instead of clicking through the
per-target ``plots/`` directories.

Run with:
    uv run marimo edit afpert/visualize/rmsf_notebook.py     # interactive
    uv run marimo run  afpert/visualize/rmsf_notebook.py     # read-only app
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr

    sns.set_theme(style="whitegrid")

    def fig_to_png(fig, dpi: int = 110):
        """Rasterise a figure to a PNG mo.image.

        marimo embeds matplotlib figures as SVG by default; the per-residue
        scatter has one vector point per residue, so the SVG blows past the
        output size limit. A raster PNG keeps each plot small and bounded.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return mo.image(buf.getvalue())

    # Inlined from afpert.visualize.rmsf so the notebook needs no afpert import
    # (marimo doesn't put the repo root on sys.path). Names must NOT start with
    # "_" or marimo treats them as cell-private and won't share them across cells.
    def rmsf_corr(df):
        """Pearson/Spearman of RMSF vs experimental displacement over matched residues."""
        paired = df.dropna(subset=["rmsf", "displacement"])
        if len(paired) >= 2:
            r, rp = pearsonr(paired["rmsf"], paired["displacement"])
            rho, rhop = spearmanr(paired["rmsf"], paired["displacement"])
        else:
            r = rp = rho = rhop = float("nan")
        return paired, r, rp, rho, rhop

    def ci_band(df):
        """Lower/upper RMSF band, falling back to the point estimate if no CI columns."""
        lo = df["rmsf_lo"] if "rmsf_lo" in df else df["rmsf"]
        hi = df["rmsf_hi"] if "rmsf_hi" in df else df["rmsf"]
        return lo, hi

    return Path, ci_band, fig_to_png, mo, np, pd, plt, rmsf_corr


@app.cell
def _(Path, mo):
    # repo root = three levels up from afpert/visualize/rmsf_notebook.py
    RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"

    entries = sorted(
        (p.parent.parent.name, p.parent.name, p)  # (target, method, csv path)
        for p in RUNS_ROOT.glob("*/*/rmsf_scores.csv")
    )
    targets = sorted({t for t, _, _ in entries})
    methods = sorted({m for _, m, _ in entries})

    mo.md(
        f"**{len(entries)}** per-residue RMSF plot(s) found under `{RUNS_ROOT}` "
        f"— {len(targets)} target(s), {len(methods)} method(s)."
    )
    return entries, methods, targets


@app.cell
def _(methods, mo, targets):
    target_sel = mo.ui.multiselect(
        options=targets, value=targets, label="Targets"
    )
    method_sel = mo.ui.multiselect(
        options=methods, value=methods, label="Methods"
    )
    mo.hstack([target_sel, method_sel], justify="start", gap=2)
    return method_sel, target_sel


@app.cell
def _(ci_band, np, pd, plt, rmsf_corr):
    def rmsf_figure(target: str, method: str, csv):
        df = pd.read_csv(csv)
        paired, r, rp, rho, rhop = rmsf_corr(df)
        lo, hi = ci_band(df)

        fig, (ax_line, ax_sc) = plt.subplots(
            1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 1]}
        )

        c_rmsf, c_disp = "tab:blue", "tab:red"
        ax_line.fill_between(
            df["residue"], lo, hi, color=c_rmsf, alpha=0.25, lw=0,
            label="RMSF 95% CI",
        )
        ax_line.plot(df["residue"], df["rmsf"], color=c_rmsf, lw=1.5, label="RMSF")
        ax_line.set_xlabel("Residue")
        ax_line.set_ylabel("RMSF (Å)", color=c_rmsf)
        ax_line.tick_params(axis="y", labelcolor=c_rmsf)

        if df["displacement"].notna().any():
            ax_disp = ax_line.twinx()
            ax_disp.grid(False)
            ax_disp.plot(
                df["residue"], df["displacement"], color=c_disp, lw=1.5,
                alpha=0.8, label="Cα displacement",
            )
            ax_disp.set_ylabel("Experimental Cα displacement (Å)", color=c_disp)
            ax_disp.tick_params(axis="y", labelcolor=c_disp)

        n_pred = int(df["n_predictions"].iloc[0]) if "n_predictions" in df else len(df)
        ax_line.set_title(
            f"{target} / {method}  (n={n_pred} predictions)"
        )

        # correlation scatter (only meaningful where displacement is known)
        if not paired.empty:
            ax_sc.scatter(
                paired["displacement"], paired["rmsf"],
                c=paired["residue"], cmap="viridis",
                edgecolor="black", s=40, alpha=0.8,
            )
            if len(paired) >= 2:
                b, a = np.polyfit(paired["displacement"], paired["rmsf"], 1)
                xs = np.array([paired["displacement"].min(), paired["displacement"].max()])
                ax_sc.plot(xs, a + b * xs, color="black", ls="--", lw=1, alpha=0.6)
            ax_sc.set_xlabel("Experimental Cα displacement (Å)")
            ax_sc.set_ylabel("RMSF (Å)")
            ax_sc.set_title(
                f"Pearson r = {r:.2f} (p={rp:.1e})\n"
                f"Spearman ρ = {rho:.2f} (p={rhop:.1e})"
            )
        else:
            ax_sc.set_axis_off()
            ax_sc.text(0.5, 0.5, "no displacement data", ha="center", va="center")

        fig.tight_layout()
        return fig

    return (rmsf_figure,)


@app.cell
def _():
    return


@app.cell
def _(entries, fig_to_png, method_sel, mo, rmsf_figure, target_sel):
    tset, mset = set(target_sel.value), set(method_sel.value)
    selected = [(t, m, p) for t, m, p in entries if t in tset and m in mset]

    blocks = [fig_to_png(rmsf_figure(t, m, p)) for t, m, p in selected]

    mo.vstack(blocks) if blocks else mo.md("_No plots match the current filter._")
    return


if __name__ == "__main__":
    app.run()
