#!/usr/bin/env python3
"""
make all perturb combos for grid of (q, m) for a target with prediction subdirs per depth

query and msa masks come from independent rng streams
if msa mask positions overlap query mask positions, reject to keep disjoint
depths are colab runtime arg so leave blank for now but make dirs

Layout:

<target>/
    <method>/
        original.a3m
        <target>_<q>_<m>_<seed>/
            <target>_<q>_<m>_<seed>.a3m
            predictions/
                directories for the different depths
"""

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable, List

from afpert.io.a3m import read_a3m, write_a3m
from afpert.perturb import perturb_disjoint, perturb_alphamask

PERTURB_METHODS = {
    "afsample2": perturb_disjoint,
    "alphamask": perturb_alphamask,
}


def parse_int_list(specs: str) -> List[int]:
    nums: List[int] = []
    for w in specs.split(","):
        if w.strip():
            nums.append(int(w))
    return nums


def generate_runs(base_a3m: Path, target: str, query_masks: Iterable[int], msa_masks: Iterable[int], depths: Iterable[int], num_seeds: int, runs_root: Path, method: str = "afsample2", overwrite: bool = False) -> List[Path]:

    if method not in PERTURB_METHODS:
        raise ValueError(f"unknown method {method!r}, choose from {sorted(PERTURB_METHODS)}")
    perturb_fn = PERTURB_METHODS[method]

    a3m_entries = read_a3m(base_a3m)
    target_root = runs_root / target / method
    target_root.mkdir(parents=True, exist_ok=True)

    copy_original = target_root / "original.a3m"
    if overwrite or not copy_original.exists():
        write_a3m(a3m_entries, copy_original)

    written: List[Path] = []
    skipped_exists: List[Path] = []
    depths = list(depths)

    for q, m in product(query_masks, msa_masks):

        n_seeds = 1 if (q == 0 and m == 0) else num_seeds

        for seed in range(n_seeds):

            run_name = f"{target}_{q}_{m}_{seed}"
            run_dir = target_root / run_name
            a3m_path = run_dir / f"{run_name}.a3m"

            if a3m_path.exists() and not overwrite:
                skipped_exists.append(a3m_path)
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                perturbed = perturb_fn(a3m_entries, q / 100.0, m / 100.0, seed)
                write_a3m(perturbed, a3m_path)
                written.append(a3m_path)

            for d in depths:
                (run_dir / "predictions" / f"{d}").mkdir(parents=True, exist_ok=True)

    print(f"wrote {len(written)} perturbed a3m files under {target_root}/")
    if skipped_exists:
        print(f"didnt overwrite {len(skipped_exists)} files, --overwrite not set")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("original_a3m", type=Path, help="path to base (unperturbed) a3m")
    p.add_argument("target", help="target name, used as run-dir prefix")
    p.add_argument("--query-masks", type=parse_int_list, required=True, help="comma-separated query mask percentages, e.g. 0,15,30")
    p.add_argument("--msa-masks", type=parse_int_list, required=True, help="comma-separated MSA column mask percentages, e.g. 0,15,30")
    p.add_argument("--depths", type=parse_int_list, required=True, help="comma-separated --max-msa depths for prediction subdirs, e.g. 32,256,1024")
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--method", choices=sorted(PERTURB_METHODS), default="afsample2", help="perturbation backend: afsample2 (random disjoint) or alphamask (coevolution-ranked)")
    p.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = p.parse_args()

    generate_runs(
        base_a3m=args.original_a3m,
        target=args.target,
        query_masks=args.query_masks,
        msa_masks=args.msa_masks,
        depths=args.depths,
        num_seeds=args.num_seeds,
        runs_root=args.runs_root,
        method=args.method,
        overwrite=args.overwrite,
    )
