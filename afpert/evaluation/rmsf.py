import glob
from pathlib import Path

import biotite.structure as struct
import biotite.structure.io
import numpy as np
import pandas as pd
from biotite.structure.superimpose import superimpose, superimpose_homologs
from tqdm import tqdm


def _load_ref_ca(name: str) -> struct.AtomArray:
    """Load a reference structure's CA atoms, tolerant to how the name is passed.

    The evaluate pipeline passes bare ids (e.g. ``6IRS_B``) and the existing
    build_*_db helpers prepend ``../``; we try that plus a couple of obvious
    fallbacks so this works whether the refs sit next to or one level above the
    project root.
    """
    candidates = [name, f"{name}.pdb", f"../{name}.pdb", f"./{name}.pdb"]
    for cand in candidates:
        if Path(cand).exists():
            arr = biotite.structure.io.load_structure(cand)
            return arr[arr.atom_name == "CA"]
    raise FileNotFoundError(
        f"could not find reference structure for {name!r} (tried {candidates})"
    )


def _rmsf_from_fitted(fitted: np.ndarray) -> np.ndarray:
    """Per-residue RMSF for an (n_pred, n_res, 3) stack already in a common frame."""
    mean_coord = fitted.mean(axis=0)
    sq_dev = ((fitted - mean_coord) ** 2).sum(axis=2)  # (n_pred, n_res)
    return np.sqrt(sq_dev.mean(axis=0))  # (n_res,)


def _bootstrap_rmsf_ci(
    fitted: np.ndarray, n_boot: int, ci: tuple[float, float], seed: int
):
    """Bootstrap CI for the per-residue RMSF by resampling ensemble members.

    The RMSF is an ensemble statistic, so its sampling uncertainty is estimated by
    resampling predictions with replacement and recomputing the full statistic
    (each resample uses its own mean structure). Returns ``(lo, hi)`` arrays at the
    requested percentiles, in the same residue order as ``fitted``.
    """
    n_pred = fitted.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, fitted.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_pred, size=n_pred)
        boot[b] = _rmsf_from_fitted(fitted[idx])
    lo, hi = np.percentile(boot, [ci[0], ci[1]], axis=0)
    return lo, hi


PARAM_COLS = ["MSA Samples", "Query Mask %", "MSA Mask %"]


def _parse_params(path: str) -> dict:
    parts = Path(path).name.split("_")
    return {
        "MSA Samples": int(path.split("predictions")[1].split("/")[1]),
        "Query Mask %": int(parts[1]),
        "MSA Mask %": int(parts[2]),
    }


def _rmsf_frame(fitted, residue, res_name, disp, n_boot, ci, seed):
    """Per-residue RMSF + bootstrap CI + displacement for one ensemble -> DataFrame."""
    rmsf = _rmsf_from_fitted(fitted)
    rmsf_lo, rmsf_hi = _bootstrap_rmsf_ci(fitted, n_boot, ci, seed)
    return pd.DataFrame(
        {
            "residue": residue,
            "res_name": res_name,
            "rmsf": np.around(rmsf, 3),
            "rmsf_lo": np.around(rmsf_lo, 3),
            "rmsf_hi": np.around(rmsf_hi, 3),
            "displacement": np.around(disp, 3),
            "n_predictions": fitted.shape[0],
        }
    )


def build_rmsf_db(
    structure_1,
    structure_2,
    target_dir,
    method="afsample2",
    n_boot: int = 1000,
    ci: tuple[float, float] = (2.5, 97.5),
    seed: int = 0,
    per_combo: bool = True,
):
    """Per-residue RMSF over the prediction ensemble vs. experimental Ca displacement.

    Per-residue quantities written to ``rmsf_scores.csv`` so they can be plotted
    and correlated:

    * ``rmsf`` -- positional fluctuation of each Ca across *all* predictions for
      ``target_dir``/``method`` (same glob as build_rmsd_db/build_tm_db). All
      predictions share the target sequence, so they are superimposed directly
      onto a common frame (refined once onto the ensemble mean) and the RMSF is
      computed in prediction-residue numbering.
    * ``rmsf_lo`` / ``rmsf_hi`` -- bootstrap confidence interval for ``rmsf`` at the
      ``ci`` percentiles (``n_boot`` resamples of the ensemble members). Shows how
      stable the per-residue RMSF is given the ensemble size, so method-to-method
      differences can be read as real or within sampling noise.
    * ``displacement`` -- distance between the two reference conformations
      (``structure_1`` vs ``structure_2``) at the matching Ca, i.e. the
      experimental conformational change the perturbation is meant to recover.

    The two are aligned by mapping each prediction residue to a reference
    residue with biotite's sequence-aware ``superimpose_homologs``; residues with
    no reference match get ``NaN`` displacement.

    With ``per_combo`` (default), the same RMSF + bootstrap CI is also computed for
    each perturbation combination ``(MSA Samples, Query Mask %, MSA Mask %)`` that
    has predictions and written long-format to ``rmsf_by_combo.csv``. Each combo's
    ensemble is superimposed onto the *pooled* mean frame so the curves are directly
    comparable, and the per-residue ``displacement`` is identical across combos.
    """
    ref1 = _load_ref_ca(structure_1)
    ref2 = _load_ref_ca(structure_2)

    fitted_ref2, _, _, _ = superimpose_homologs(ref1, ref2)
    _, _, ref1_anchor, ref2_anchor = superimpose_homologs(
        ref1, ref2, outlier_threshold=np.inf
    )
    disp_by_resid: dict[int, float] = {}
    for r1, r2 in zip(ref1_anchor, ref2_anchor):
        d = float(np.linalg.norm(ref1.coord[r1] - fitted_ref2.coord[r2]))
        disp_by_resid[int(ref1.res_id[r1])] = d

    all_pdbs = glob.glob(
        f"runs/{target_dir}/{method}/*/predictions/*/*_unrelaxed_rank_*.pdb"
    )

    # collect the ensemble of CA coordinate sets. All predictions are the target
    # sequence, so they share residue ordering; keep only those whose length
    # matches the first one (guards against the odd truncated model). Each kept
    # member carries its perturbation params so combos can be split out below.
    ensemble = []
    params = []
    template = None
    for i in tqdm(all_pdbs, desc="loading predictions"):
        mobile = biotite.structure.io.load_structure(i)
        ca = mobile[mobile.atom_name == "CA"]
        if template is None:
            template = ca.copy()
        if ca.array_length() != template.array_length():
            print(f"skipping {i}: {ca.array_length()} CA != {template.array_length()}")
            continue
        ensemble.append(ca)
        params.append(_parse_params(i))

    # superimpose every member onto the first, take the mean structure, then
    # refine by superimposing onto that mean (one pass is plenty for CA RMSF).
    fitted0 = np.stack([superimpose(template, m)[0].coord for m in ensemble], axis=0)
    mean_struct = template.copy()
    mean_struct.coord = fitted0.mean(axis=0)
    fitted = np.stack( [superimpose(mean_struct, m)[0].coord for m in ensemble], axis=0 )  # (n_pred, n_res, 3)

    # fill relevant positions with res ids from the superimposed s
    _, _, ref1_idx, pred_idx = superimpose_homologs(ref1, template, outlier_threshold=np.inf)
    disp_per_pred = np.full(template.array_length(), np.nan)  # start with nans
    for r1, p in zip(ref1_idx, pred_idx):
        d = disp_by_resid.get(int(ref1.res_id[r1]))
        if d is not None: # so if not left out after the superimposing of ref1 with ref2
            disp_per_pred[p] = d  # fill in

    residue = template.res_id.astype(int)
    res_name = template.res_name

    base = Path(f"runs/{target_dir}/{method}")
    base.mkdir(parents=True, exist_ok=True)

    df = _rmsf_frame(fitted, residue, res_name, disp_per_pred, n_boot, ci, seed)
    out = base / "rmsf_scores.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(ensemble)} predictions, {len(df)} residues)")

    if per_combo:
        params_df = pd.DataFrame(params)
        combo_frames = []
        for key, idx in params_df.groupby(PARAM_COLS, sort=True).groups.items():
            members = np.asarray(idx)
            sub = _rmsf_frame(
                fitted[members], residue, res_name, disp_per_pred, n_boot, ci, seed
            )
            for col, val in zip(PARAM_COLS, key):
                sub[col] = val
            combo_frames.append(sub)
        combo_df = pd.concat(combo_frames, ignore_index=True)
        combo_out = base / "rmsf_by_combo.csv"
        combo_df.to_csv(combo_out, index=False)
        print(f"wrote {combo_out} ({len(combo_frames)} combos x {len(df)} residues)")
