#!/usr/bin/env python3
"""
loop over all target a3ms in msas/ and run the perturb grid for each

use like this
  python -m afpert.scripts.cross_perturb_all --query-masks 0,15,30 --msa-masks 0,15,30 --depths 32,256,1024

each <target>.a3m in --msas-dir becomes one target named <target>.
"""

import argparse
from pathlib import Path

from afpert.scripts.cross_perturb_target import generate_runs, parse_int_list


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets-dir", type=Path, default=Path("msas"), help="dir of folders (per target), containing another folder, that contains a uniref.a3m file")
    p.add_argument("--query-masks", type=parse_int_list, required=True, help="comma-separated query mask percentages, e.g. 0,15,30")
    p.add_argument("--msa-masks", type=parse_int_list, required=True, help="comma-separated MSA column mask percentages, e.g. 0,15,30")
    p.add_argument("--depths", type=parse_int_list, required=True, help="comma-separated --max-msa depths, e.g. 32,256,1024")
    p.add_argument("--mode", choices=["independent", "disjoint"], default="independent", help="masking strategy (see cross_perturb_target docstring)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = p.parse_args()

    a3ms = []
    for item in Path(args.msa_dir).iterdir():
        if item.is_dir():
            for item2 in item.iterdir():
                if item2.is_file() and item2.match("uniref.a3m"):
                    new_path = item2.with_name(f"{item.name}.a3m")
                    item2.rename(new_path)
                    a3ms.append(new_path)

    if not a3ms:
        raise SystemExit(f"no *.a3m files found in {args.msas_dir}/")

    print(f"found {len(a3ms)} target a3ms in {args.msas_dir}/")
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
            mode=args.mode,
            overwrite=args.overwrite,
        )
