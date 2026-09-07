#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from afpert.visualize import benchmark, benchmarkdumbell, rmsd, rmsf, tm, bestopenclosed, boxplots

# afpert/scripts/visualize.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Plot perturbation scores.")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--rmsf", action="store_true", help="plot RMSF (pooled, per-combo, and cross-method)")
    parser.add_argument("--rmsf-aggregate", action="store_true", help="cross-target RMSF effect boxplot over all runs")
    parser.add_argument("--benchmark", action="store_true", help="plot all per-combination benchmark heatmaps")
    parser.add_argument("--dumbell", action="store_true", help="plot dumbbell overview")
    parser.add_argument("--panels", action="store_true", help="best open vs best closed panels across all methods")
    parser.add_argument("--boxplot", action="store_true", help="best open vs best closed boxplot")
    parser.add_argument("--target", required=True, help="target name (e.g. 7DSQ_2)")
    parser.add_argument("--xlabel", type=str, help="")
    parser.add_argument("--ylabel", type=str, help="")
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom"], default="afsample2", help="perturbation method subtree to plot")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.tm:
        tm.plot_all(args.method, args.target, args.xlabel, args.ylabel)
    if args.rmsd:
        rmsd.plot_all(args.method, args.target, args.xlabel, args.ylabel)
    if args.rmsf:
        rmsf.plot_all(args.method, args.target)
        rmsf.compare_per_residue_rmsf(args.target)
    if getattr(args, "rmsf_aggregate", False):
        rmsf.aggregate()
    if args.benchmark:
        benchmark.plot_all(args.method, args.target)
    if args.dumbell:
        benchmarkdumbell.plot_all(args.method, args.target, args.xlabel, args.ylabel)
    if args.target and args.panels and args.xlabel and args.ylabel:
        bestopenclosed.panels(args.target, args.xlabel, args.ylabel)
    if args.boxplot:
        boxplots.plot()



if __name__ == "__main__":
    main()
