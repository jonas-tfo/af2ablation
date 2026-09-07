from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tmtools import tm_align

from afpert.io.pdb import get_residue_data, load_pdb


Guide = Tuple[str, np.ndarray, str]


def first_chain_data(pdb_path: Path ) -> Tuple[np.ndarray, str]:
    """
    Returns:
        tuple of ca coords and sequence for first chain of pdb
    """
    return get_residue_data(next(load_pdb(pdb_path).get_chains()))


def tm_score(mobile_pdb: Path, target_pdb: Path) -> Dict[str, float]:
    """
    tm align of mobile against target (first chains of each)
    tm_norm_chain1 is normalized by the mobile length, tm_norm_chain2 by the target length
    """
    m_coords, m_seq = first_chain_data(mobile_pdb)
    t_coords, t_seq = first_chain_data(target_pdb)
    res = tm_align(m_coords, t_coords, m_seq, t_seq)
    return {
        "rmsd": res.rmsd,
        "tm_norm_chain1": res.tm_norm_chain1,
        "tm_norm_chain2": res.tm_norm_chain2,
    }


def load_guide(pdb_path: Path) -> Guide:
    """pre load a reference structure as (name, coords, seq)"""
    coords, seq = first_chain_data(pdb_path)
    return (Path(pdb_path).stem, coords, seq)


def tm_against_guides(mobile_pdb: Path, guides: List[Guide]) -> Dict[str, float]:
    """Return {guide_name: tm_norm_chain1} for mobile vs each pre-loaded guide."""
    m_coords, m_seq = first_chain_data(mobile_pdb)
    return {
        name: tm_align(m_coords, gc, m_seq, gs).tm_norm_chain1
        for name, gc, gs in guides
    }
