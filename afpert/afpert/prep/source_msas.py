#!/usr/bin/env python3

"""
get msas for all target fasta files from a directory
"""

import subprocess
from pathlib import Path


def get_msas(target_dir = Path("targets/"), msas_dir = Path("msas/")):
    for target in target_dir.iterdir():
        if target.is_file() and str(target).endswith(".fasta"):
            output_dir = msas_dir / str(target).rstrip(".fasta")
            cmd = ["/home/friedrich/localcolabfold/.pixi/envs/default/bin/colabfold_batch", "--msa-only", target, output_dir]
            subprocess.run(cmd, check=True)
