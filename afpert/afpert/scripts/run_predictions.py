#!/usr/bin/env python3
"""
run colabfold_batch on every <target>_q<q>_m<m>.a3m under runs/ for each depth.
run with 
python -m afpert.scripts.run_predictions
"""

import subprocess
import sys
from pathlib import Path

# 1. is the dir names 2. is the flag to use for depth
DEPTHS = [
    ("16", "8:16"),
    ("32", "16:32"),
    ("256", "128:256"),
    ("1024", "512:1024"),
    ("5120", None),  # full MSA
]


def run(a3m: Path, out_dir: Path, max_msa: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/home/friedrich/localcolabfold/.pixi/envs/default/bin/colabfold_batch",
        "--num-recycle", "1",
        "--num-models", "5",
        "--model-type", "alphafold2",
        "--model-order", "1,2,3,4,5",
        "--random-seed", "0",
        "--num-seeds", "5",
    ]
    if max_msa:
        cmd += ["--max-msa", max_msa]
    cmd += [str(a3m), str(out_dir)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    runs_root = Path("runs")
    for run_dir in sorted(runs_root.glob("*/*_*_*")):
        a3m = run_dir / f"{run_dir.name}.a3m"
        if not a3m.exists():
            continue
        for sub_name, max_msa in DEPTHS:
            run(a3m, run_dir / "predictions" / sub_name, max_msa)