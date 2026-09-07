#!/usr/bin/env python3

"""
get msas for all target fasta files from a directory
"""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGETS_DIR = PROJECT_ROOT / "targets"
MSA_DIR = PROJECT_ROOT / "msas"

def get_msas(target_dir = TARGETS_DIR, msas_dir = MSA_DIR):
    for target in target_dir.iterdir():
        if target.is_file() and str(target).endswith(".fasta"):
            output_dir = msas_dir / str(target).rstrip(".fasta")
            cmd = ["/home/friedrich/localcolabfold/.pixi/envs/default/bin/colabfold_batch", "--msa-only", target, output_dir]
            subprocess.run(cmd)

get_msas()