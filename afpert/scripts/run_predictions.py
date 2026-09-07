#!/usr/bin/env python3
"""
run colabfold_batch on every <target>_<q>_<m>_<s>.a3m under runs/ for each depth.
run with 
python -m afpert.scripts.run_predictions
"""

import subprocess
import sys
from pathlib import Path
import argparse
from afpert.scripts.cross_perturb_target import parse_int_list

# 1. is the dir names 2. is the flag to use for depth
# DEPTHS = [
#     ("16", "8:16"),
#     ("32", "16:32"),
#     ("256", "128:256"),
#     ("1024", "512:1024"),
#     ("5120", None),  # full MSA
# ]


def run(a3m: Path, out_dir: Path, max_msa: str | None, overwrite = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/home/friedrich/localcolabfold/.pixi/envs/default/bin/colabfold_batch",
        "--num-recycle", "1",
        "--num-models", "5",
        "--model-type", "alphafold2",
        "--model-order", "1,2,3,4,5",
        "--random-seed", "0",
        "--num-seeds", "1",
    ]
    if max_msa:
        cmd += ["--max-msa", max_msa]
    if overwrite:
        cmd += ["--overwrite-existing-results"]
    cmd += [str(a3m), str(out_dir)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":


    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=parse_int_list, required=True, help="comma-separated --max-msa depths, e.g. 32,256,1024")
    p.add_argument("--target-dir", type=str, required=True, help="")
    p.add_argument("--method", type=str, required=True, help="")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    save_depths = []

    depths = list(args.depths)

    for depth in depths:
        if depth == 5120:
            save_depths.append( ("5120", None)  ) 
        else:
            save_depths.append((f"{depth}", f"{depth // 2}:{depth}"))


    runs_root = Path("runs") / args.target_dir / args.method
    # layout: runs/<target>/<method>/<target>_<q>_<m>_<seed>/
    for run_dir in sorted(runs_root.glob("*_*_*_*")):
        print(run_dir)
        a3m = run_dir / f"{run_dir.name}.a3m"
        if not a3m.exists():
            continue
        for sub_name, max_msa in save_depths:
            run(a3m, run_dir / "predictions" / sub_name, max_msa, args.overwrite)
