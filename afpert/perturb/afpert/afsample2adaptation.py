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

def perturb_a3m(entries: List[Tuple[str, str]], mask_fraction: float, seed):
    if len(entries) == 0:
        print("No entries parsed")
    rng = np.random.default_rng(seed)
    query_seq = entries[0][1]
    n_cols = len(query_seq)
    n_mask = int(n_cols * mask_fraction)
    masked_cols = set(rng.choice(n_cols, size=n_mask, replace=False).tolist())

    perturbed_entries: List[Tuple[str, str]] = [entries[0]] # dont mask query

    for header, seq in entries[1:]:
        out: List[str] = []
        col: int = 0
        for ch in seq:
            if ch.islower():
                # insertions, not part of MSA, leave but dont count as column
                out.append(ch)
            else:
                if col in masked_cols:
                    out.append("-")
                else:
                    out.append(ch)
                col += 1
        perturbed_entries.append((header, "".join(out)))
    return perturbed_entries

if __name__ == "__main__":
    a3m_path: Path = Path.cwd() / Path("data") / "tst.a3m"
    output_dir: Path = Path.cwd() / Path("outputs")
    entries = read_a3m(a3m_path)
    for i in tqdm(range(10)):
        perturbed = perturb_a3m(entries, 0.15, i)
        output_path = output_dir / f"perturbed_{i:03d}.custom.a3m"
        write_a3m(perturbed, output_path)

