#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from afpert.visualize import rmsd, tm

# afpert/scripts/visualize.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Plot perturbation scores.")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--target", required=True, help="target name (e.g. 7DSQ_2)")
    parser.add_argument("--method", choices=["afsample2", "alphamask"], default="afsample2", help="perturbation method subtree to plot")
    args = parser.parse_args()

    # plot functions read CSVs at runs/<method>/<target>/... relative to the project root
    os.chdir(PROJECT_ROOT)

    if args.tm:
        tm.plot_all(args.method, args.target)
    if args.rmsd:
        rmsd.plot_all(args.method, args.target)


if __name__ == "__main__":
    main()
