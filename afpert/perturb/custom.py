import numpy as np
from typing import List, Tuple
import glob

def perturb(entries: List[Tuple[str, str]], query_fraction: float, msa_fraction: float, seed: int, target:str) -> List[Tuple[str, str]]:

    arrs = []
    for i in glob.glob(f"./runs/{target}/processed/{target}/*.npy"):
        arr = np.load(i)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    
        arrs.append(arr)
    
    alldata = np.array(arrs)
    alldata = np.log(np.percentile(alldata, axis=0, q=90) / (np.percentile(alldata, axis=0, q=50) + 1e-12))
    
    ls = ([{int(x%len(alldata)), int(x)//len(alldata)} for x in np.argsort(alldata, axis=None) if (x%len(alldata) < x//len(alldata))])
    
    bestcontacts = (ls[::-1][:len(alldata)])

    if not entries:
        return entries

    n_cols = len(entries[0][1])

    n_q = int(n_cols * query_fraction)
    n_m = int(n_cols * msa_fraction)

    rng = np.random.default_rng(seed)
    choice = rng.choice(bestcontacts, size=int(n_m/2))

    msa_mask_positions: set = set().union(*choice)

    query_mask_positions: set = set().union(rng.choice(list(set().union([x for x in range(n_cols)]) - msa_mask_positions), n_q))

    # query mask
    header, query = entries[0]
    new_query_chars: List[str] = []
    for i, ch in enumerate(query):
        if i in query_mask_positions:
            new_query_chars.append("A")
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