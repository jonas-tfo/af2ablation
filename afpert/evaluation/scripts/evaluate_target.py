#!/usr/bin/env python3

from afpert.evaluation.tm import build_tm_db
from afpert.evaluation.rmsd import build_rmsd_db
from pathlib import Path
import argparse
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--target-dir", type=str, help="dir to save the csv to")
    parser.add_argument("--structure1", type=str, help="struct 1")
    parser.add_argument("--structure2", type=str, help="struct 2")
    parser.add_argument("--method", choices=["afsample2", "alphamask"], default="afsample2", help="perturbation method subtree to evaluate")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    struct1 = args.structure1
    struct2 = args.structure2
    target_dir = args.target_dir

    if args.tm:
        build_tm_db(struct1, struct2, target_dir, method=args.method)
    if args.rmsd:
        build_rmsd_db(struct1, struct2, target_dir, method=args.method)
