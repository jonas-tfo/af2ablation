"""
Coevolution-guided masking, ported from roccomoretti/alphamask (MSAUtils.get_coevolution).

Picks columns to mask by Direct Coupling Analysis (DCA) score instead of uniform random.
Drop-in replacement for afsample2.perturb_disjoint: same signature, same a3m semantics
(uppercase columns are aligned, lowercase are insertions and pass through).
"""

from typing import List, Tuple
import numpy as np

_RESTYPES = "ARNDCQEGHILKMFPSTWYV"
_RESTYPES_X_GAP = _RESTYPES + "X-"
_AA_TO_IDX = {c: i for i, c in enumerate(_RESTYPES_X_GAP)}
_GAP_IDX = _AA_TO_IDX["-"]
_X_IDX = _AA_TO_IDX["X"]


def _is_aligned_col(c: str) -> bool:
    """A3M aligned column: uppercase A-Z or gap '-'. Lowercase is insertion (skip); whitespace/digits/stray chars are also skipped to tolerate slightly dirty files."""
    return c == "-" or ("A" <= c <= "Z")


def _entries_to_array(entries: List[Tuple[str, str]]) -> np.ndarray:
    """
    Convert a3m entries to (n_seq, n_cols) integer array over _RESTYPES_X_GAP.
    Only aligned columns (uppercase or '-') are kept; insertions and stray chars skipped.
    """
    n_cols = sum(1 for c in entries[0][1] if _is_aligned_col(c))
    arr = np.full((len(entries), n_cols), _GAP_IDX, dtype=np.int8)
    for row, (header, seq) in enumerate(entries):
        col = 0
        for ch in seq:
            if not _is_aligned_col(ch):
                continue
            if col >= n_cols:
                raise ValueError(
                    f"row {row} has more aligned columns than query ({n_cols}); "
                    f"header={header!r}"
                )
            arr[row, col] = _AA_TO_IDX.get(ch, _X_IDX)
            col += 1
    return arr


def coevolution_scores(msa_array: np.ndarray) -> np.ndarray:
    """
    DCA coupling matrix with shrinkage + APC. Returns (L, L) numpy array.
    """
    n_seq, n_cols = msa_array.shape
    n_classes = len(_RESTYPES_X_GAP)

    one_hot = np.eye(n_classes, dtype=np.float32)[msa_array]
    flat = one_hot.reshape(n_seq, -1)

    cov = np.cov(flat.T)
    shrink = (4.5 / np.sqrt(n_seq)) * np.eye(cov.shape[0], dtype=cov.dtype)
    prec = np.linalg.inv(cov + shrink)

    diag = np.diag(prec)
    pcorr = prec / np.sqrt(np.outer(diag, diag))

    reshaped = pcorr.reshape(n_cols, n_classes, n_cols, n_classes)
    coup = np.sqrt(np.square(reshaped[:, :20, :, :20]).sum(axis=(1, 3)))

    idx = np.arange(n_cols)
    coup[idx, idx] = 0.0

    row_sum = coup.sum(axis=0, keepdims=True)
    col_sum = coup.sum(axis=1, keepdims=True)
    total = coup.sum()
    if total > 0:
        coup = coup - (row_sum * col_sum) / total
    coup[idx, idx] = 0.0
    return coup


def _select_top_columns(scores: np.ndarray, n: int, seed: int) -> set:
    """
    Pick the n columns with the highest per-position coupling strength.
    Ties broken by a seeded RNG so different seeds give different picks among equals.
    """
    if n <= 0:
        return set()
    per_col = scores.sum(axis=0)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0, 1e-9, size=per_col.shape)
    order = np.argsort(-(per_col + jitter))
    return {int(x) for x in order[:n]}


def perturb_disjoint(
    entries: List[Tuple[str, str]],
    query_fraction: float,
    msa_fraction: float,
    seed: int,
) -> List[Tuple[str, str]]:
    """
    Query columns = top n_q coupled positions.
    MSA columns   = next-highest coupled positions not already in the query set.
    Seed only breaks ties selection is otherwise deterministic for a fixed MSA.
    """
    if not entries:
        return entries

    n_cols = sum(1 for c in entries[0][1] if _is_aligned_col(c))
    n_q = int(n_cols * query_fraction)
    n_m = int(n_cols * msa_fraction)

    if n_q == 0 and n_m == 0:
        return [(h, s) for h, s in entries]

    arr = _entries_to_array(entries)
    scores = coevolution_scores(arr)
    per_col = scores.sum(axis=0)

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0, 1e-9, size=per_col.shape)
    order = np.argsort(-(per_col + jitter))

    query_mask_positions = {int(x) for x in order[:n_q]}
    msa_candidates = [int(x) for x in order if int(x) not in query_mask_positions]
    msa_mask_positions = set(msa_candidates[:n_m])

    header, query = entries[0]
    new_query_chars: List[str] = []
    for i, ch in enumerate(query):
        if i in query_mask_positions:
            new_query_chars.append("A")
        else:
            new_query_chars.append(ch)
    out: List[Tuple[str, str]] = [(header, "".join(new_query_chars))]

    for header, seq in entries[1:]:
        new_chars: List[str] = []
        col = 0
        for ch in seq:
            if ch.islower():
                new_chars.append(ch)
            else:
                if col in msa_mask_positions:
                    new_chars.append("X")
                else:
                    new_chars.append(ch)
                col += 1
        out.append((header, "".join(new_chars)))

    return out
