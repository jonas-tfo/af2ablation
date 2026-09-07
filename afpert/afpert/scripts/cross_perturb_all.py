#!/usr/bin/env python3
"""
loop over all target a3ms in targets/<target>/<subdir>/<target>.a3m|uniref.a3m and do perturb stuff

<targets-dir>/
    <target>/
        <subdir>/
            uniref.a3m

uniref.a3m is renamed in place to <target>.a3m, then used as the base a3m for the perturb grid.

use like this
python -m afpert.scripts.cross_perturb_all --query-masks 0,15,30 --msa-masks 0,15,30 --depths 32,256,1024
"""

import argparse
from pathlib import Path

from afpert.scripts.cross_perturb_target import generate_runs, parse_int_list


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets-dir", type=Path, default=Path("targets"), help="dir containing <target>/<subdir>/uniref.a3m for each target")
    p.add_argument("--query-masks", type=parse_int_list, required=True, help="comma-separated query mask percentages, e.g. 0,15,30")
    p.add_argument("--msa-masks", type=parse_int_list, required=True, help="comma-separated MSA column mask percentages, e.g. 0,15,30")
    p.add_argument("--depths", type=parse_int_list, required=True, help="comma-separated --max-msa depths, e.g. 32,256,1024")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = p.parse_args()


    a3ms = []
    for target_dir in sorted(p for p in args.targets_dir.iterdir() if p.is_dir()):
        target = target_dir.name
        # will be uniref.a3m for first run and after that is renamed to <target>.a3m, so look for both
        matches = list(target_dir.glob(f"*/{target}.a3m")) or list(target_dir.glob("*/uniref.a3m"))
        if not matches:
            continue
        path = matches[0]
        if path.name == "uniref.a3m":
            path = path.rename(path.with_name(f"{target}.a3m"))
        a3ms.append(path)

    if not a3ms:
        raise SystemExit(f"no */*/uniref.a3m files found under {args.targets_dir}/")

    print(f"found {len(a3ms)} target a3ms under {args.targets_dir}/")
    for a3m in a3ms:
        target = a3m.stem
        print(f"\n-- {target} --")
        generate_runs(
            base_a3m=a3m,
            target=target,
            query_masks=args.query_masks,
            msa_masks=args.msa_masks,
            depths=args.depths,
            seed=args.seed,
            runs_root=args.runs_root,
            overwrite=args.overwrite,
        )