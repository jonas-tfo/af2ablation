from typing import List, Tuple
import numpy as np


def perturb_independent(entries: List[Tuple[str, str]], query_fraction: float, msa_fraction: float, seed: int) -> List[Tuple[str, str]]:
    """
    Query and MSA column masks are drawn from two independent RNG streams spawned from `seed`.

    Within each stream the mask is cumulative: for a fixed seed, the positions chosen at
    fraction=0.30 are a strict superset of those chosen at fraction=0.15. So walking the
    (query_fraction, msa_fraction) grid keeps the query mask identical while msa grows,
    and vice versa — "freeze query mask, layer on msa masking" is well defined.

    Query and msa masks may overlap; disjointness is not enforced.
    """
    if not entries:
        return entries

    n_cols = len(entries[0][1])
    n_q = int(n_cols * query_fraction)
    n_m = int(n_cols * msa_fraction)

    ss = np.random.SeedSequence(seed)
    rng_q, rng_m = (np.random.default_rng(s) for s in ss.spawn(2))

    query_perm = rng_q.permutation(n_cols)
    msa_perm = rng_m.permutation(n_cols)

    query_mask_positions = {int(x) for x in query_perm[:n_q]}
    msa_mask_positions = {int(x) for x in msa_perm[:n_m]}

    header, query = entries[0]
    new_query_chars: List[str] = []
    for i, ch in enumerate(query):
        if i in query_mask_positions:
            new_query_chars.append("X")
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


def perturb(entries: List[Tuple[str, str]], query_fraction: float, msa_fraction: float, seed: int) -> List[Tuple[str, str]]:
    """
    Apply query and MSA column masking keeping the two mask sets as disjoint as possible
    """
    if not entries:
        return entries

    n_cols = len(entries[0][1])

    n_q = int(n_cols * query_fraction)
    n_m = int(n_cols * msa_fraction)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_cols)

    query_mask_positions: set = set()
    for i in range(n_q):
        query_mask_positions.add(int(shuffled[i]))

    msa_mask_positions: set = set()
    for i in range(n_m):
        # modulo to wrap around for minimal overlap
        wrapped_i = (n_q + i) % n_cols
        msa_mask_positions.add(int(shuffled[wrapped_i]))

    # query mask
    header, query = entries[0]
    new_query_chars: List[str] = []
    for i, ch in enumerate(query):
        if i in query_mask_positions:
            new_query_chars.append("X")
        else:
            new_query_chars.append(ch)
    out: List[Tuple[str, str]] = [(header, "".join(new_query_chars))]

    # msa mask
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

def perturb_disjoint(entries: List[Tuple[str, str]], query_fraction: float, msa_fraction: float, seed: int) -> List[Tuple[str, str]]:
    """
    makes the msa mask into the first positions (according to the fraction) that are not already in the query mask


    Same as perturb_independent but rejects msa columns that overlap with the query mask.
    The msa mask is taken as the first n_m positions of the msa permutation that are NOT
    already in the query mask.

    Cumulative-in-msa property is preserved for a fixed query_fraction. When query_fraction
    grows, some previously-eligible msa positions get knocked out, so the msa mask shifts —
    that's the price of disjointness.

    If query_fraction + msa_fraction > 1, the msa mask is silently capped at what's left.
    """
    if not entries:
        return entries

    n_cols = len(entries[0][1])
    n_q = int(n_cols * query_fraction)
    n_m = int(n_cols * msa_fraction)

    ss = np.random.SeedSequence(seed)
    rng_q, rng_m = (np.random.default_rng(s) for s in ss.spawn(2))

    query_perm = rng_q.permutation(n_cols)
    msa_perm = rng_m.permutation(n_cols)

    query_mask_positions = {int(x) for x in query_perm[:n_q]}
    msa_candidates = [int(x) for x in msa_perm if int(x) not in query_mask_positions]
    msa_mask_positions = set(msa_candidates[:n_m])

    header, query = entries[0]
    new_query_chars: List[str] = []
    for i, ch in enumerate(query):
        if i in query_mask_positions:
            new_query_chars.append("X")
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
