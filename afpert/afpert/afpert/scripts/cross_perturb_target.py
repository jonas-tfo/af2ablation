#!/usr/bin/env python3
"""
make all perturb combos for grid of (q%, m%, d) for one target

use like this
  python -m afpert.scripts.cross_perturb_target <base.a3m> <target> --query-masks 0,15,30 --msa-masks 0,15,30 --depths 32,256,1024

mask modes:
  independent (default) — query and msa masks come from independent RNG streams and are
    cumulative within each stream. holding q fixed and varying m keeps the query mask byte
    identical across runs (and vice versa), so you can compare "query-only" vs
    "query + msa" predictions cleanly.
  disjoint — legacy behavior: one shared permutation, query/msa positions chosen to overlap
    as little as possible.

Layout:

<target>/
    original.a3m
    <target>_q<q>_m<m>_d<d>/
        <target>_q<q>_m<m>_d<d>.a3m
        predictions/
            colabfold output files
"""

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable, List

from afpert.io.a3m import read_a3m, write_a3m
from afpert.perturb import perturb, perturb_independent


# TODO try to handle list arg parsing better
def parse_int_list(specs: str) -> List[int]:
    nums: List[int] = []
    for w in specs.split(","):
        if w.strip():
            nums.append(int(w))
    return nums


def generate_runs(base_a3m: Path, target: str, query_masks: Iterable[int], msa_masks: Iterable[int], depths: Iterable[int], seed: int, runs_root: Path, mode: str = "independent", overwrite: bool = False) -> List[Path]:

    if mode == "independent":
        perturb_fn = perturb_independent
    elif mode == "disjoint":
        perturb_fn = perturb
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'independent' or 'disjoint'")

    a3m_entries = read_a3m(base_a3m)
    target_root = runs_root / target
    target_root.mkdir(parents=True, exist_ok=True)

    copy_original = target_root / "original.a3m"
    if overwrite or not copy_original.exists():
        write_a3m(a3m_entries, copy_original)

    written: List[Path] = []
    skipped_exists: List[Path] = []

    for q, m, d in product(query_masks, msa_masks, depths):
        run_name = f"{target}_q{q}_m{m}_d{d}"
        run_dir = target_root / run_name
        out_path = run_dir / f"{run_name}.a3m"

        # TODO maybe just always overwrite shouldnt matter
        if out_path.exists() and not overwrite:
            skipped_exists.append(out_path)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        perturbed = perturb_fn(a3m_entries, q / 100.0, m / 100.0, seed)
        write_a3m(perturbed, out_path)
        written.append(out_path)

    print(f"wrote {len(written)} perturbed a3m files under {target_root}/ (mode={mode})")
    if skipped_exists:
        print(f"didnt overwrite {len(skipped_exists)} files, --overwrite not set")

    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("original_a3m", type=Path, help="path to base (unperturbed) a3m")
    p.add_argument("target", help="target name, used as run-dir prefix")
    p.add_argument("--query-masks", type=parse_int_list, required=True, help="comma-separated query mask percentages, e.g. 0,15,30")
    p.add_argument("--msa-masks", type=parse_int_list, required=True, help="comma-separated MSA column mask percentages, e.g. 0,15,30")
    p.add_argument("--depths", type=parse_int_list, required=True, help="comma-separated --max-msa depths, e.g. 32,256,1024")
    p.add_argument("--mode", choices=["independent", "disjoint"], default="independent", help="masking strategy (see module docstring)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = p.parse_args()

    generate_runs(
        base_a3m=args.original_a3m,
        target=args.target,
        query_masks=args.query_masks,
        msa_masks=args.msa_masks,
        depths=args.depths,
        seed=args.seed,
        runs_root=args.runs_root,
        mode=args.mode,
        overwrite=args.overwrite,
    )

