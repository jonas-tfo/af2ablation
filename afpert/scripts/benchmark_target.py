#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from afpert.evaluation.benchmark import summarize_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, help="target dir under runs/")
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom"], default="afsample2")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    summary = summarize_run(args.target_dir, args.method)
    if summary is None:
        raise SystemExit(f"no tm_scores.csv under runs/{args.target_dir}/{args.method}/")
    print(summary.to_string(index=False))
