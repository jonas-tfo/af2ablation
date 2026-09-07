#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from afpert.visualize import benchmark, rmsd, tm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot perturbation scores for every target under runs/<method>/.")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--benchmark", action="store_true", help="plot all per-combination benchmark heatmaps")
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom"], default="afsample2", help="perturbation method subtree to plot")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    method_root = Path(f"runs/7BCQ_1/{args.method}")
    if not method_root.is_dir():
        raise SystemExit(f"no method tree at {method_root}/")

    targets = sorted(p.name for p in method_root.iterdir() if p.is_dir())
    if not targets:
        raise SystemExit(f"no targets found under {method_root}/")

    print(f"plotting {len(targets)} target(s) under {method_root}/")
    for target in targets:
        target = "7BCQ_1"
        print(f"-- {target} --")
        if args.tm:
            tm.plot_all(args.method, target)
        if args.rmsd:
            rmsd.plot_all(args.method, target)
        if args.benchmark:
            benchmark.plot_all(args.method, target)
