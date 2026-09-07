#!/usr/bin/env python3
"""Scrollable overview of every per-target RMSF-compare plot.

Discovers all ``runs/<target>/rmsf_compare.png`` files and renders them inline,
so the method-comparison plots (produced by
``afpert.visualize.rmsf.compare_per_residue_rmsf``) can be scrolled through in a
single notebook instead of opening each target's file by hand.

Run with:
    uv run marimo edit afpert/visualize/rmsf_compare_notebook.py     # interactive
    uv run marimo run  afpert/visualize/rmsf_compare_notebook.py     # read-only app
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    # repo root = three levels up from afpert/visualize/rmsf_compare_notebook.py
    RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"

    # one rmsf_compare.png per target: runs/<target>/rmsf_compare.png
    entries = sorted(
        (p.parent.name, p) for p in RUNS_ROOT.glob("*/rmsf_compare.png")
    )
    targets = [t for t, _ in entries]
    return RUNS_ROOT, entries, mo, targets


@app.cell
def _(RUNS_ROOT, mo, targets):
    mo.md(
        f"**{len(targets)}** RMSF-compare plot(s) found under `{RUNS_ROOT}`."
        if targets
        else f"**No `rmsf_compare.png` found under `{RUNS_ROOT}`.**\n\n"
        "Generate them with "
        "`afpert.visualize.rmsf.compare_per_residue_rmsf(target)`."
    )
    return


@app.cell
def _(mo, targets):
    target_sel = mo.ui.multiselect(
        options=targets, value=targets, label="Targets"
    )
    target_sel if targets else None
    return (target_sel,)


@app.cell
def _(entries, mo, target_sel):
    tset = set(target_sel.value)
    selected = [(t, p) for t, p in entries if t in tset]

    blocks = []
    for target, png in selected:
        blocks.append(mo.md(f"### {target}"))
        blocks.append(mo.image(png.read_bytes()))

    mo.vstack(blocks) if blocks else mo.md("_No plots match the current filter._")
    return


if __name__ == "__main__":
    app.run()
