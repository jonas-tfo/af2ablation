#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from afpert.evaluation.benchmark import benchmark_all

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom"], action="append",
                        help="restrict to this method (repeatable); default: all")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    benchmark_all(methods=args.method)
