#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

def read_a3m(path: Path) -> List[Tuple[str, str]]:
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
    with open(path, "w") as f:
        for header, seq in entries:
            f.write(f"{header}\n{seq}\n")

def get_aligned_length(seq: str) -> int:
    count: int = 0
    for ch in seq:
        if ch.isupper() or ch == "-":
            count += 1
    return count

# columns masked in the query and columns masked in the MSA need to be disjoint
def disjoint_masks(n_cols, query_fraction, msa_fraction, seed):
    """
    Pick two index sets over [0, n_cols) that are disjoint when possible
    """
    rng = np.random.default_rng(seed)
    n_q = int(n_cols * query_fraction)
    n_m = int(n_cols * msa_fraction)

    if n_q + n_m <= n_cols:
        perm = rng.permutation(n_cols)
        return set(perm[:n_q].tolist()), set(perm[n_q:n_q + n_m].tolist())

    # fallback to independent sampling if not possible, may have some overlap
    q = set(rng.choice(n_cols, size=n_q, replace=False).tolist())
    m = set(rng.choice(n_cols, size=n_m, replace=False).tolist())
    return q, m


def perturb_msa(entries, mask_fraction, seed, masked_cols=None, token="-"):
    query_seq = entries[0][1]
    n_cols = len(query_seq)
    if masked_cols is None:
        rng = np.random.default_rng(seed)
        n_mask = int(n_cols * mask_fraction)
        masked_cols = set(rng.choice(n_cols, size=n_mask, replace=False).tolist())

    perturbed_entries = [entries[0]]
    for header, seq in entries[1:]:
        out, col = [], 0
        for ch in seq:
            if ch.islower():
                out.append(ch)
            else:
                out.append(token if col in masked_cols else ch)
                col += 1
        perturbed_entries.append((header, "".join(out)))
    return perturbed_entries


def perturb_query(entries, mask_fraction, seed, masked_cols=None, mask_token="X") -> List[Tuple[str, str]]:
    header, query = entries[0]
    n_cols = len(query)
    if masked_cols is None:
        rng = np.random.default_rng(seed)
        n_mask = int(n_cols * mask_fraction)
        masked_cols = set(rng.choice(n_cols, size=n_mask, replace=False).tolist())
    new_query = ""
    for i, ch in enumerate(query):
        if i in masked_cols:
            new_query += mask_token
        else:
            new_query += ch
    return [(header, new_query)] + entries[1:]


if __name__ == "__main__":
    # args parse to make subprocess work
    # args: msa_mask_fract, query_mask_fract, a3m_path (or also fasta path), seed (maybe)
    a3m_path: Path = Path.cwd() / Path("data") / "tst.a3m"
    output_dir: Path = Path.cwd() / Path("outputs")
    entries = read_a3m(a3m_path)
    mask_ratio = 30

    for i in tqdm(range(10)):
        perturbed = perturb_msa(entries, mask_ratio/100, i)
        output_path = output_dir / f"perturbed_{i:03d}_{mask_ratio}.custom.a3m"
        write_a3m(perturbed, output_path)

        # print(perturb_msa(entries, 0.15, seed=i)) # just msa

        entry = perturb_query(entries=entries, mask_fraction=0.50, seed=i) # just query
        print(entry[0][1])

        # Combined, disjoint
        q_idx, m_idx = disjoint_masks(n_cols=len(entries[0][1]), query_fraction=0.15, msa_fraction=0.15, seed=i)
        out = perturb_query(entries=entries, mask_fraction=0, seed=i, masked_cols=q_idx)
        out = perturb_msa(entries=out, mask_fraction=0, seed=i, masked_cols=m_idx)
