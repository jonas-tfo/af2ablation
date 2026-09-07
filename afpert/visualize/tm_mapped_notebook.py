#!/usr/bin/env python3
"""Scrollable overview of every TM-mapped (AFsample2 Fig. 5 style) plot.

Discovers all ``runs/<target>/<method>/tm_scores.csv`` files and renders the
TM(to open) vs TM(to closed) scatter for each target inline: the full ensemble
as a grey cloud, with predictions that match a reference conformation
(max of the two TM-scores >= threshold) highlighted and coloured by that best
TM-score. Use the method dropdown to switch between afsample2 and custom.

Run with:
    uv run marimo edit afpert/visualize/tm_mapped_notebook.py     # interactive
    uv run marimo run  afpert/visualize/tm_mapped_notebook.py     # read-only app
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import functools
    import io
    from pathlib import Path
    import itertools
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1 as THREE_TO_ONE
    from tmtools import tm_align
    import seaborn as sns

    sns.set_context("talk") # Options: "paper", "notebook", "talk", "poster"
    canonicals = ["custom", "alphamask", "afsample2"]

    map_names = {
        "custom": "Custom (ours)",
        "AFSample2":"AFSample2",
        "afvanilla": "Vanilla AF2",
        "msasubsampling": "MSA Subsampling",
        "SPEACH_AF": "SPEACH_AF",
        "afsample": "AFsample",
        "afsample2": "afsample2"
    }

    def fig_to_png(fig, dpi: int = 110):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return mo.image(buf.getvalue())

    def ca_coords_seq(pdb_path: str):
        structure = PDBParser(QUIET=True).get_structure("g", pdb_path)
        coords, seq = [], []
        for res in structure[0].get_residues():
            if "CA" in res:
                coords.append(res["CA"].coord)
                seq.append(THREE_TO_ONE.get(res.resname, "X"))
        return np.array(coords), "".join(seq)

    @functools.lru_cache(maxsize=None)
    def guide_cross_tm(mobile_pdb: str, fixed_pdb: str) -> float:
        cm, sm = ca_coords_seq(mobile_pdb)
        cf, sf = ca_coords_seq(fixed_pdb)
        return float(tm_align(cm, cf, sm, sf).tm_norm_chain2)

    def guide_points(xlabel: str, ylabel: str, pdbs_root: Path):
        open_pdb = pdbs_root / "open" / f"{xlabel}.pdb"
        closed_pdb = pdbs_root / "closed" / f"{ylabel}.pdb"
        print(xlabel, ylabel)
        if not (open_pdb.exists() and closed_pdb.exists()):
            return None
        c_open = guide_cross_tm(str(open_pdb), str(closed_pdb))   # open vs closed
        c_closed = guide_cross_tm(str(closed_pdb), str(open_pdb))  # closed vs open
        return (1.0, c_open), (c_closed, 1.0)

    def mapped_figure(target: str, method: str | list, csv, threshold: float, pdbs_root: Path):
        rows = 1
        fig, ax = plt.subplots(figsize=(16, 5), nrows= rows if isinstance(method, list) else 1, ncols=(len(method))//rows if isinstance(method, list) else 1)
        fig.suptitle(target)
        if isinstance(csv, str):
            csv = [csv]
        gp = None
        xlabel, ylabel = "open", "closed"
        cbar_ax = fig.add_axes((0.95, 0.15, 0.02, 0.7))

        for i in range(len(csv)):
            df = pd.read_csv(csv[i])
            if method[i] in canonicals:
                tm_cols = [c for c in df.columns if c.endswith("_tm")]
                if len(tm_cols) < 2:
                    return None
                xcol, ycol = tm_cols[0], tm_cols[1]
                xlabel, ylabel = xcol[:-3], ycol[:-3]
                gp = guide_points(xlabel, ylabel, pdbs_root)
                if threshold == 1.0:
                    threshold = round(gp[0][1], 2)
            else:
                xcol = "TM_open"
                ycol = "TM_close"

            x, y = df[xcol], df[ycol]
            best = df[[xcol, ycol]].max(axis=1)
            mask = best >= threshold

            # full ensemble
            ax.flatten()[i].set_aspect(1)
            ax.flatten()[i].scatter(x, y, s=25, color="darkslategrey", alpha=0.6, linewidths=0, zorder=1)
            sc = ax.flatten()[i].scatter(
                x[mask], y[mask],
                c=best[mask], cmap="viridis_r",
                s=90, edgecolor="black", linewidth=0.6, zorder=2,
                vmax=1, vmin=threshold,
            )
            if mask.any():
                cbar = fig.colorbar(sc, cbar_ax)
                cbar.set_label("TM-score of model to reference")

            # guide conformations

            if i > 0:
                ax.flatten()[i].set_yticks([])
            if gp is not None:
                (ox, oy), (cx, cy) = gp
                ax.flatten()[i].scatter([ox], [oy], marker="*", s=420, color="red", edgecolor="black", linewidth=0.8, zorder=4, label=f"{xlabel} guide")
                ax.flatten()[i].scatter([cx], [cy], marker="*", s=420, color="royalblue", edgecolor="black", linewidth=0.8, zorder=4, label=f"{ylabel} guide")
                ax.flatten()[i].axhline(oy, color="#0065bd", ls="--")
                ax.flatten()[i].axvline(cx, color="#0065bd", ls="--")
                #ax[i//2][i%2].legend(loc="lower left", framealpha=0.9)
                lo = max(min(min(ox, oy) - 0.1, float(np.quantile(x,0.75))-0.10), 0.3)
                fig.subplots_adjust(right=0.8)
            else:
                lo = max(min(x.min(), y.min()) - 0.03, 0.4)
            ax.flatten()[i].set_xlim(lo, 1.02)
            ax.flatten()[i].set_ylim(lo, 1.02)
            fig.supxlabel(f"TM-score (to {xlabel})")
            fig.supylabel(f"TM-score (to {ylabel})")
            ax.flatten()[i].set_title(f"{map_names[method[i]]}", fontweight=500, fontsize=22) # : {int(mask.sum())} match(es) with TM ≥ {threshold}
        fig.tight_layout()

        fig.subplots_adjust(right=0.92)
        return fig

    return Path, fig_to_png, mapped_figure, mo


@app.cell
def _(Path, mo):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    RUNS_ROOT = REPO_ROOT / "runs"
    PDBS_ROOT = REPO_ROOT / "pdbs"  # pdbs/open/<id>.pdb, pdbs/closed/<id>.pdb

    entries = sorted(
        (p.parent.parent.name, p.parent.name, p)  # (target, method, csv path)
        for p in RUNS_ROOT.glob("*/*/tm_scores.csv")
    )

    entries.extend(sorted(
        (p.parent.name, p.parent.parent.name, p)  # (target, method, csv path)
        for p in REPO_ROOT.glob("afsample_paper_models/generated_models/oc23/*/*/final_df_tmalign.csv")
    ))

    entries.extend(sorted(
        (p.parent.name, "AFSample2", p)  # (target, method, csv path)
        for p in REPO_ROOT.glob("afsample_paper_models/generated_models/oc23/afsample2/af_io_abl_15/*/final_df_tmalign.csv")
    ))


    targets = sorted({t for t, _, _ in entries})
    methods = sorted({m for _, m, _ in entries})

    mo.md(
        f"**{len(entries)}** TM-mapped plot(s) found under `{RUNS_ROOT}` "
        f"— {len(targets)} target(s), {len(methods)} method(s). "
        f"Guide markers use references under `{PDBS_ROOT}`"
        f"{' (not found — markers hidden)' if not PDBS_ROOT.exists() else ''}."
    )
    return PDBS_ROOT, entries, methods, targets


@app.cell
def _(methods, mo, targets):
    method_sel = mo.ui.multiselect(
        options=methods,
        value=["custom", "AFSample2", "msasubsampling", "afvanilla"] if "custom" in methods else ([methods[0]] if methods else None),
        label="Method",
    )
    threshold_sel = mo.ui.slider(
        start=0.5, stop=1.0, step=0.01, value=0.9, label="Threshold", show_value=True
    )
    target_sel = mo.ui.multiselect(options=targets, value=[targets[-1]], label="Targets")
    mo.hstack([method_sel, threshold_sel, target_sel], justify="start", gap=2)
    return method_sel, target_sel, threshold_sel


@app.cell
def _(
    PDBS_ROOT,
    entries,
    fig_to_png,
    mapped_figure,
    method_sel,
    mo,
    target_sel,
    threshold_sel,
):
    tset = set(target_sel.value)
    selected = [
        (t, m, p) for t, m, p in entries if m in method_sel.value and t in tset
    ]

    selecteddict = {}

    for i in selected:
        if i[0]  not in selecteddict.keys():
            selecteddict[i[0]] = ([], [])
        selecteddict[i[0]][0].append(i[1])
        selecteddict[i[0]][1].append(i[2])



    blocks = []
    for t in selecteddict.keys():
        fig = mapped_figure(t, selecteddict[t][0], selecteddict[t][1], threshold_sel.value, PDBS_ROOT)
        if fig is None:
            blocks.append(mo.md(f"_{t} / {selecteddict[t][0][0]}: fewer than two `*_tm` columns, skipped._"))
        else:
            blocks.append(fig_to_png(fig))


    mo.vstack(blocks) if blocks else mo.md("_No plots match the current filter._")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
