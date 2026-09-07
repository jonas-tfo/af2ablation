#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple


def read_a3m(path: Path) -> List[Tuple[str, str]]:
    """
    returns a list of tuples, each containing a header the corresponding sequence
    """
    entries: List[Tuple[str, str]] = []
    header: str | None = None
    seq: str = ""

    for line in path.read_text().splitlines():
        # not needed for monomer
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, seq))
            header = line
            seq = ""
        else:
            seq += line

    if header is not None:
        entries.append((header, seq))

    return entries


def write_a3m(entries: List[Tuple[str, str]], path):
    """
    input is a list of tuples, each containing a header the corresponding sequence
    """
    with open(path, "w") as f:
        for header, seq in entries:
            f.write(f"{header}\n{seq}\n")


