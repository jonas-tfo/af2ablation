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

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1 as THREE_TO_ONE
    from tmtools import tm_align
    import seaborn as sns

    sns.set_context("talk")

    REPO_ROOT = Path(__file__).resolve().parents[2]
    RUNS_ROOT = REPO_ROOT / "runs"
    RUNS_BU = REPO_ROOT /"runs_backup_150726"
    PDBS_ROOT = REPO_ROOT / "pdbs"  # pdbs/open/<id>.pdb, pdbs/closed/<id>.pdb

    OC23_ROOT = REPO_ROOT / "afsample_paper_models" / "generated_models" / "oc23"
    OC23_df = pd.read_csv(REPO_ROOT/"oc23_limited.csv")

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
        if not (open_pdb.exists() and closed_pdb.exists()):
            return None
        c_open = guide_cross_tm(str(open_pdb), str(closed_pdb))   # open vs closed
        c_closed = guide_cross_tm(str(closed_pdb), str(open_pdb))  # closed vs open
        print(xlabel, ylabel, c_open)
        return (1.0, c_open), (c_closed, 1.0)

    def closest_point_on_line(line_point1, line_point2, point):
        # Convert points to numpy arrays for vector operations
        A = np.array(line_point1)
        B = np.array(line_point2)
        P = np.array(point)

        # Calculate vectors AB and AP
        AB = B - A
        AP = P - A

        # Calculate the projection of AP onto AB
        # Parametric representation: P + t*AB
        t = np.dot(AP, AB) / np.dot(AB, AB)

        # Clamp t to the line segment [A, B]
        t = max(0, min(1, t))

        # Calculate the closest point on the line
        closest_point = A + t * AB

        return closest_point, t

    def mapped_figure(target: str, method: str, csv, threshold: float, pdbs_root: Path):
        df = pd.read_csv(csv)
        tm_cols = [c for c in df.columns if c.endswith("_tm") or c.startswith("TM_")]
        if len(tm_cols) < 2:
            return None
        xcol, ycol = tm_cols[0], tm_cols[1]
        if tm_cols[0].endswith("_tm"):
            xlabel, ylabel = xcol[:-3], ycol[:-3]

        else: xlabel, ylabel = OC23_df.loc[OC23_df["Uniprotid"] == target]["pdbid_open"].to_list()[0], OC23_df.loc[OC23_df["Uniprotid"] == target]["pdbid_closed"].to_list()[0]


        gp = guide_points(xlabel, ylabel, pdbs_root)
        x, y = df[xcol], df[ycol]
        pts = []
        fillratio = [0]
        if gp is not None:
            print(gp[0][1], gp[1][0])
            fillratio = []
            for k in range(5):
                sub_df = df.sample(500, replace=True)
                # 
                topright = sub_df[(sub_df[xcol] > gp[1][0]) & (sub_df[ycol] > gp[0][1]) & (sub_df[ycol] > gp[1][0]) & (sub_df[xcol] > gp[0][1])]
                sub_df = topright
                x_, y_ = sub_df[xcol], sub_df[ycol]
                ns = []
                for pt in zip(x_,y_):
                    ns.append(closest_point_on_line(gp[0], gp[1], pt)[1])
                    pts.append(closest_point_on_line(gp[0], gp[1], pt)[0])
                filled = np.unique([x // 0.01 for x in ns])
                bin_weights = [1 + 16 * (k/99 - 0.5)**2 for k in range(100)]
                fillratio.append(np.sum([bin_weights[int(x)] for x in filled])/np.sum(bin_weights))
            pts = np.array(pts)

        best = df[[xcol, ycol]].max(axis=1)
        mask = best >= threshold

        fig, ax = plt.subplots(figsize=(7, 5))
        # full ensemble
        if len(pts) > 0:
            ax.scatter(pts[:,0], pts[:,1], s=15, color="black", alpha=0.6, linewidths=0, zorder=0)
        ax.scatter(x, y, s=15, color="lightgrey", alpha=0.6, linewidths=0, zorder=1)
        sc = ax.scatter(
            x[mask], y[mask],
            c=best[mask], cmap="viridis_r",
            s=90, edgecolor="black", linewidth=0.6, zorder=2,
        )
        if mask.any():
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("TM-score of model to reference")

        # guide conformations
        if gp is not None:
            (ox, oy), (cx, cy) = gp
            ax.scatter([ox], [oy], marker="*", s=420, color="red", edgecolor="black", linewidth=0.8, zorder=4, label=f"{xlabel} guide")
            ax.scatter([cx], [cy], marker="*", s=420, color="royalblue", edgecolor="black", linewidth=0.8, zorder=4, label=f"{ylabel} guide")
            ax.legend(loc="lower left", framealpha=0.9)

        lo = min(x.min(), y.min()) - 0.03
        ax.set_xlim(lo, 1.02)
        ax.set_ylim(lo, 1.02)
        ax.set_xlabel(f"TMscore (to {xlabel})")
        ax.set_ylabel(f"TMscore (to {ylabel})")
        ax.set_title(f"{target} — {int(mask.sum())} match(es) with TM ≥ {threshold}")
        if fillratio is not None:
            ax.set_title(f"{target} — {int(mask.sum())} match(es) with TM ≥ {threshold}, fill-ratio mean: {np.mean(fillratio)}")
        fig.tight_layout()
        return fig, fillratio

    return (
        OC23_ROOT,
        PDBS_ROOT,
        RUNS_BU,
        RUNS_ROOT,
        fig_to_png,
        mapped_figure,
        mo,
        pd,
        plt,
        sns,
    )


@app.cell
def _(OC23_ROOT, PDBS_ROOT, RUNS_BU, RUNS_ROOT, mo):
    entries = sorted(
        (p.parent.parent.name, p.parent.name, p)  # (target, method, csv path)
        for p in RUNS_ROOT.glob("*/*/tm_scores.csv")
    )

    entries.extend(sorted(
        (p.parent.parent.name, "old", p)  # (target, method, csv path)
        for p in RUNS_BU.glob("*/custom/tm_scores.csv")
    ))

    entries.extend(sorted(
        (p.parent.name, p.parent.parent.name, p)  # (target, method, csv path)
        for p in OC23_ROOT.glob("*/*/final_df_tmalign.csv")
    ))
    entries.extend(sorted(
        (p.parent.name, "AFSample2", p)  # (target, method, csv path)
        for p in OC23_ROOT.glob("afsample2/af_io_abl_15/*/final_df_tmalign.csv")
    ))

    targets = sorted({t for t, _, _ in entries})
    methods = sorted({m for _, m, _ in entries})

    targets_limited = [
        "A2RJ53",
        "P31133",
        "P00558",
        "P40131",
        "Q7DAU8",
        "A0QTT2",
        "Q5F9M1",
        "Q18A65",
        "Q9ERE7",
        "P62495",
        "P71447",
        "Q9Z4N6",
        "A6UVT1",
        "Q53W80",
        "Q9SS90",
        "Q9X9P9",
    ]

    mo.md(
        f"**{len(entries)}** TM-mapped plot(s) found under `{RUNS_ROOT}` "
        f"— {len(targets)} target(s), {len(methods)} method(s). "
        f"Guide markers use references under `{PDBS_ROOT}`"
        f"{' (not found — markers hidden)' if not PDBS_ROOT.exists() else ''}."
    )
    return entries, methods, targets, targets_limited


@app.cell
def _(methods, mo, targets, targets_limited):
    method_sel = mo.ui.multiselect(
        options=methods,
        value=["custom",
        "old",
        "afsample",
        "SPEACH_AF",
        "msasubsampling",
        "AFSample2",
        "afvanilla", "afsample2"    
        ] if "afsample2" in methods else ([methods[0]] if methods else None),
        label="Method",
    )
    threshold_sel = mo.ui.slider(
        start=0.5, stop=1.0, step=0.01, value=0.9, label="Threshold", show_value=True
    )
    target_sel = mo.ui.multiselect(options=targets, value=targets_limited, label="Targets")
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

    blocks = []
    ls = []
    mth = method_sel.value
    meth = {}

    for n, i in enumerate(mth):
        ls.append({})
        meth[i] = n

    alldata = {"method": [], "fr": [], "target": []}
    for t, m, p in selected:
        try:
            fig, fr = mapped_figure(t, m, p, threshold_sel.value, PDBS_ROOT)
            for j in fr:
                alldata["method"].append(m)
                alldata["fr"].append(j)
                alldata["target"].append(t)
            if fig is None:
                blocks.append(mo.md(f"_{t} / {m}: fewer than two `*_tm` columns, skipped._"))
            else:
                blocks.append(fig_to_png(fig))
        except Exception as e:
            print(e)
    #mo.vstack(blocks) if blocks else mo.md("_No plots match the current filter._")
    return alldata, meth


@app.cell
def _(alldata, pd, plt, sns):
    figure, axes = plt.subplots(1, 1, figsize=(15, 6))
    figure.tight_layout()


    alldata_df = pd.DataFrame(alldata)
    # bp_data = alldata_df[alldata_df["target"] != "Q9ERE7"]
    # bp_data = bp_data[bp_data["target"] != "P62495"]
    bp_data = alldata_df

    plt.xticks(rotation=21)

    display = {
        "custom": "Custom",
        "old": "Erroneous impl",
        "afsample": "AFsample [1]",
        "SPEACH_AF": "SPEACH_AF [2]",
        "msasubsampling": "MSA Subsampling [3]",
        "AFSample2": "AFSample2 [4]",
        "afvanilla": "Vanilla AF2 [5]",
        "afsample2": "randomized masking"
    }
    order = list(display)


    cleaned_targets = [
        "A2RJ53",
        "P31133",
        "P00558",
        "P40131",
        "Q7DAU8",
        "A0QTT2",
        "Q5F9M1",
        "Q18A65",
        "Q9ERE7",
        "P62495",
        "P71447",
        "Q9Z4N6",
        "A6UVT1",
        "Q53W80",
        "Q9SS90",
        "Q9X9P9",
    ]

    # cleaned_targets.remove("P62495")
    # cleaned_targets.remove("Q9ERE7")
    sns.barplot(bp_data, errorbar="se", x = "target", y= "fr", hue="method", order=cleaned_targets, ax=axes, legend=False, hue_order=order)
    plt.xlabel("Target Protein")
    plt.ylabel("Fill-Ratio")
    axes.legend(display.values(), labelcolor=sns.color_palette().as_hex(), ncols=2, markerscale=0, handlelength=0, handletextpad=0, prop={"weight":800})
    return display, order


@app.cell
def _(alldata, display, order, pd, plt, sns):
    fg, axs = plt.subplots(1, 1, figsize=(9, 4))
    fg.tight_layout()
    sns.boxenplot(data = pd.DataFrame(alldata).groupby(["target", "method"]).agg("mean"), x="method", y="fr", hue= "method", hue_order=order, width=.6, order=order, ax=axs, showfliers=False, k_depth="full")
    sns.stripplot(data=pd.DataFrame(alldata).groupby(["target", "method"]).agg("mean"), x="method", y="fr",
                legend=False,
                jitter=0.2
                , size=8, alpha=0.7, color="black", edgecolor="black", linewidth=0.3,order=order, ax=axs
            )

    axs.axhline(0.39, color="blue", linestyle=":", linewidth=1.6, alpha=0.8, zorder=0)

    axs.set_xticks(range(len(order)))
    axs.set_xticklabels([display[m] for m in order])
    plt.ylabel("Fill-Ratio")
    plt.xlabel("Method")
    plt.xticks(rotation=21)
    # order = ["custom", "afsample [1]", "SPEACH_AF [2]", "msasubsampling [3]", "AFSample2 [4]", "afvanilla [5]"]
    return


@app.cell
def _(meth):
    meth
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
