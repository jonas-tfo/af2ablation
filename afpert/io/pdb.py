import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBIO import PDBIO
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.Structure import Structure


def load_pdb(file_path: Path, structure_id: Optional[str] = None) -> Structure:
    """
    Args:
        file_path: path to pdb 
        structure_id: identifier for the structure, defaults to file name
    Returns:
        Structure
    """
    file_path = Path(file_path)
    if structure_id is None:
        structure_id = os.path.splitext(os.path.basename(file_path))[0]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(structure_id, file_path)
    if structure is None:
        print("No structure could be read from the file " + str(file_path))
        return Structure(id=structure_id)

    return structure


def split_and_save_chains(structure: Structure, output_dir: Path = Path(".")) -> List[str]:
    """
    split structure into chains and save each as own pdb file
    Returns:
        the list of file paths written.
    """
    io = PDBIO()
    saved_paths: List[str] = []

    os.makedirs(output_dir, exist_ok=True)

    for chain in structure.get_chains():
        io.set_structure(chain)
        filename = f"{structure.get_id()}_{chain.get_id()}.pdb"
        filepath = os.path.join(str(output_dir), filename)
        io.save(filepath)
        saved_paths.append(filepath)

    return saved_paths


def get_residue_data(chain: Chain) -> Tuple[np.ndarray, str]:
    """
    Extract c alpha coordinates and the aa sequence from a chain
    """
    coords = []
    seq = []

    for residue in chain:
        if residue.id[0] != " ":
            continue
        if 'CA' in residue:
            coords.append(residue["CA"].get_coord())
            seq.append(protein_letters_3to1.get(residue.get_resname(), "X"))

    return np.array(coords), "".join(seq)


def parse_perturbation_from_name(path: Path) -> Tuple[int, int, int]:
    """
    parse (query_mask_percent, msa_column_mask_percent, msa_subsample_percent) from file name
    name needs to end with numbers like "_<a>_<b>_<c>"
    """
    parts = Path(path).stem.split("_")
    return int(parts[-3]), int(parts[-2]), int(parts[-1])


