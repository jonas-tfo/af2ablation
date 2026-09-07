#!/usr/bin/env python3

"""
get msas for all target fasta files from a directory
"""

import subprocess
from pathlib import Path
import argparse


def get_msas(target_dir = Path("targets/"), msas_dir = Path("msas/")):
    for target in target_dir.iterdir():
        if target.is_file() and str(target).endswith(".fasta"):
            output_dir = msas_dir / str(target).rstrip(".fasta")
            cmd = ["/home/friedrich/localcolabfold/.pixi/envs/default/bin/colabfold_batch", "--msa-only", target, output_dir]
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot perturbation scores for every target under runs/<method>/.")
    parser.add_argument("--targets-dir", type=str, default="targets", help="perturbation method subtree to plot")
    parser.add_argument("--msas-dir", type=str, default="msa", help="perturbation method subtree to plot")
    args = parser.parse_args()

    get_msas(args.target_dir, args.msas_dir)