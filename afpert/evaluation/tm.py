import glob
import json
from pathlib import Path

import biotite.structure as struct
import biotite.structure.io
import numpy as np
import pandas as pd
from biotite.sequence import ProteinSequence
from biotite.structure.superimpose import superimpose_homologs
from biotite.structure.tm import tm_score
from tmtools import tm_align
from tqdm import tqdm

from afpert.evaluation.incremental import load_done, merge_and_write

# def build_tm_db_old(structure_1, structure_2, target_dir, method="afsample2"):
#     """
#     Per-prediction TM-score against two guide structures, using biotite's rigid-body
#     superimpose + tm_score. Same slicing convention as build_rmsd_db so both metrics
#     align residue-for-residue across the run.

#     TM-score is normalised by the mobile (predicted) chain length, matching the
#     `tm_norm_chain1` semantics of the previous tmtools implementation.
#     """
#     guides = [
#         biotite.structure.io.load_structure(f"../{structure_2}.pdb"),
#         biotite.structure.io.load_structure(f"../{structure_1}.pdb"),
#     guides = [
#         struct.concatenate(
#             (
#                 guides[0][guides[0].atom_name == "CA"][:113],
#                 guides[0][guides[0].atom_name == "CA"][130:],
#             )
#         ),
#         guides[1][guides[1].atom_name == "CA"][4:-1],
#     ]

#     prots = {
#         "points": [[], []],
#         "msa_batch": [],
#         "mean_plddt": [],
#         "subsample_sz": [],
#         "msa_mask_fraction": [],
#         "query_mask_fraction": [],
#         "pert_seed": [],
#     }

#     # unrelaxed only: runs done with --amber also emit *_relaxed_rank_*.pdb, which
#     # would double-count each prediction and have no matching *_scores_rank_*.json
#     # under the "unrelaxed"->"scores" name swap below.
#     all_pdbs = glob.glob(
#         f"runs/{target_dir}/{method}/*/predictions/*/*_unrelaxed_rank_*.pdb"
#     )

#     for i in tqdm(all_pdbs):
#         # compute everything that can fail (e.g. missing scores json) before appending
#         # anything, so a skip never leaves the columns at unequal lengths.
#         try:
#             mean_plddt = np.mean(
#                 json.load(
#                     open(i.replace("unrelaxed", "scores").replace(".pdb", ".json"))
#                 )["plddt"]
#             )
#         except:
#             print(f"warning, skipping {i}")
#             continue

#         mobile_struct = biotite.structure.io.load_structure(i)
#         mobile_struct = struct.concatenate(
#             (
#                 mobile_struct[mobile_struct.atom_name == "CA"][46:159],
#                 mobile_struct[mobile_struct.atom_name == "CA"][176:488],
#             )
#         )

#         prots["msa_batch"].append(0)
#         prots["subsample_sz"].append(int(i.split("predictions")[1].split("/")[1]))
#         prots["mean_plddt"].append(mean_plddt)
#         prots["query_mask_fraction"].append(int(i.split("/")[-1].split("_")[2]))
#         prots["msa_mask_fraction"].append(int(i.split("/")[-1].split("_")[3]))
#         prots["pert_seed"].append(int(i.split("/")[-1].split("_")[4]))

#         for j in range(len(guides)):
#             # biotite's tm_score does NOT superimpose; it scores the coordinates as
#             # given. So the mobile must be fitted onto *this* guide before scoring,
#             # otherwise (e.g. fitting once onto guides[0]) the other guide's TM is
#             # measured from the wrong pose and comes out systematically too low.
#             n = min(guides[j].array_length(), mobile_struct.array_length())
#             idx = np.arange(n)
#             fitted, _ = struct.superimpose(guides[j][idx], mobile_struct[idx])
#             tm = struct.tm_score(
#                 guides[j][idx],
#                 fitted,
#                 idx,
#                 idx,
#                 reference_length=mobile_struct.array_length(),
#             )
#             prots["points"][j].append(float(tm))

#     prt = {
#         "7DSQ_tm": prots["points"][1],  # points[1] = 7DSQ guide
#         "6IRS_tm": prots["points"][0],  # points[0] = 6IRS guide
#         "msa_batch": prots["msa_batch"],
#         "mean_plddt": prots["mean_plddt"],
#         "MSA Samples": [int(x) for x in prots["subsample_sz"]],
#         "Query Mask %": prots["query_mask_fraction"],
#         "MSA Mask %": prots["msa_mask_fraction"],
#         "MSA Mask %":   prots["msa_mask_fraction"],
#         "Perturbation Seed": prots["pert_seed"],
#     }
#     df_tm = pd.DataFrame(prt)
#     df_tm.to_csv(f"runs/{target_dir}/{method}/tm_scores.csv")


def three2one(ca: struct.AtomArray):
    out = []
    for res_name in ca.res_name:
        try:
            out.append(ProteinSequence.convert_letter_3to1(res_name))
        except KeyError:
            out.append("X")
    return "".join(out)

#def build_tm_db(structure_1, structure_2, target_dir, method="afsample2"):

def _peptide_ca(arr):
    """CA atoms of amino-acid residues only.

    biotite's superimpose_homologs / tm_score require peptide-only structures,
    so a plain ``atom_name == "CA"`` filter is not enough: it also keeps calcium
    ions (PDB element CA) and any non-standard residue with a CA atom, which makes
    biotite raise "Reference structure must be peptide only".
    """
    return arr[struct.filter_amino_acids(arr) & (arr.atom_name == "CA")]


def build_tm_db(structure_1, structure_2, target_dir, method="afsample2", incremental=False):
    out_csv = f"runs/{target_dir}/{method}/tm_scores.csv"
    prev_df, done = load_done(out_csv, incremental)

    ref2 = biotite.structure.io.load_structure(f"../{structure_2}.pdb")
    ref1 = biotite.structure.io.load_structure(f"../{structure_1}.pdb")
    guides = [_peptide_ca(ref2), _peptide_ca(ref1)]

    prots = {
        "points": [[], []],
        "path": [],
        "msa_batch": [],
        "mean_plddt": [],
        "subsample_sz": [],
        "msa_mask_fraction": [],
        "query_mask_fraction": [],
        "pert_seed": [],
    }

    all_pdbs = glob.glob(
        f"runs/{target_dir}/{method}/*/predictions/*/*_unrelaxed_rank_*.pdb"
    )
    all_pdbs = [i for i in all_pdbs if i not in done]
    if incremental:
        print(f"{target_dir}/{method}: scoring {len(all_pdbs)} new prediction(s), "
              f"{len(done)} already in {out_csv}")

    n_fail = 0
    for i in tqdm(all_pdbs):
        try:
            mean_plddt = np.mean(
                json.load(
                    open(i.replace("unrelaxed", "scores").replace(".pdb", ".json"))
                )["plddt"]
            )
        except:
            print(f"warning, skipping {i}")
            continue

        mobile_struct = biotite.structure.io.load_structure(i)
        mobile_ca = _peptide_ca(mobile_struct)

        # score both guides first; only commit to the columns if both succeed, so
        # one bad prediction can't abort the whole target or desync the lists.
        try:
            tms = []
            for guide in guides:
                fitted, _, fixed_anchor_idx, mobile_anchor_idx = (
                    struct.superimpose_homologs(guide, mobile_ca)
                )
                tms.append(float(struct.tm_score(
                    guide, fitted, fixed_anchor_idx, mobile_anchor_idx,
                    reference_length=mobile_ca.array_length(),
                )))
        except Exception as e:
            n_fail += 1
            print(f"skipping {i}: {e}")
            continue

        prots["points"][0].append(tms[0])
        prots["points"][1].append(tms[1])
        prots["path"].append(i)
        prots["msa_batch"].append(0)
        prots["subsample_sz"].append(int(i.split("predictions")[1].split("/")[1]))
        prots["mean_plddt"].append(mean_plddt)
        prots["query_mask_fraction"].append(int(i.split("/")[-1].split("_")[1]))
        prots["msa_mask_fraction"].append(int(i.split("/")[-1].split("_")[2]))
        prots["pert_seed"].append(int(i.split("/")[-1].split("_")[3]))

    if n_fail:
        print(f"{target_dir}/{method}: {n_fail} prediction(s) failed scoring")

    if '/' in structure_1:
        structure_1 = structure_1.split("/")[-1]

    if '/' in structure_2:
        structure_2 = structure_2.split("/")[-1]
    prt = {
        f"{structure_1}_tm": prots["points"][1],
        f"{structure_2}_tm": prots["points"][0],
        "path": prots["path"],
        "msa_batch": prots["msa_batch"],
        "mean_plddt": prots["mean_plddt"],
        "MSA Samples": [int(x) for x in prots["subsample_sz"]],
        "Query Mask %": prots["query_mask_fraction"],
        "MSA Mask %": prots["msa_mask_fraction"],
        "Perturbation Seed": prots["pert_seed"],
    }
    df_tm = pd.DataFrame(prt)
    merge_and_write(prev_df, df_tm, out_csv)


def build_tm_db_tmalign(
    structure_1: str, structure_2: str, target_dir: str, method: str = "afsample2",
    incremental: bool = False,
):
    out_csv = f"runs/{target_dir}/{method}/tm_scores.csv"
    prev_df, done = load_done(out_csv, incremental)

    ref2 = biotite.structure.io.load_structure(f"../{structure_2}.pdb")
    ref1 = biotite.structure.io.load_structure(f"../{structure_1}.pdb")
    guides = [ref2[ref2.atom_name == "CA"], ref1[ref1.atom_name == "CA"]]
    guide_seqs = [three2one(g) for g in guides]

    prots = {
        "points": [[], []],
        "path": [],
        "msa_batch": [],
        "mean_plddt": [],
        "subsample_sz": [],
        "msa_mask_fraction": [],
        "query_mask_fraction": [],
        "pert_seed": [],
    }

    all_pdbs = glob.glob(
        f"runs/{target_dir}/{method}/*/predictions/*/*_unrelaxed_rank_*.pdb"
    )
    all_pdbs = [i for i in all_pdbs if i not in done]
    if incremental:
        print(f"{target_dir}/{method}: scoring {len(all_pdbs)} new prediction(s), "
              f"{len(done)} already in {out_csv}")

    for i in tqdm(all_pdbs):
        try:
            mean_plddt = np.mean(
                json.load(
                    open(i.replace("unrelaxed", "scores").replace(".pdb", ".json"))
                )["plddt"]
            )
        except:
            print(f"warning, skipping {i}")
            continue

        mobile_struct = biotite.structure.io.load_structure(i)
        mobile_ca = mobile_struct[mobile_struct.atom_name == "CA"]
        mobile_seq = three2one(mobile_ca)

        prots["path"].append(i)
        prots["msa_batch"].append(0)
        prots["subsample_sz"].append(int(i.split("predictions")[1].split("/")[1]))
        prots["mean_plddt"].append(mean_plddt)
        prots["query_mask_fraction"].append(int(i.split("/")[-1].split("_")[1]))
        prots["msa_mask_fraction"].append(int(i.split("/")[-1].split("_")[2]))
        prots["pert_seed"].append(int(i.split("/")[-1].split("_")[3]))

        for guide_idx in range(len(guides)):
            res = tm_align(
                x=mobile_ca.coord,
                y=guides[guide_idx].coord,
                seqx=mobile_seq,
                seqy=guide_seqs[guide_idx],
            )
            prots["points"][guide_idx].append(float(res.tm_norm_chain2))

    if '/' in structure_1:
        structure_1 = structure_1.split("/")[-1]
    
    if '/' in structure_2:
        structure_2 = structure_2.split("/")[-1]

    prt = {
        f"{structure_1}_tm": prots["points"][1],
        f"{structure_2}_tm": prots["points"][0],
        "path": prots["path"],
        "msa_batch": prots["msa_batch"],
        "mean_plddt": prots["mean_plddt"],
        "MSA Samples": [int(x) for x in prots["subsample_sz"]],
        "Query Mask %": prots["query_mask_fraction"],
        "MSA Mask %": prots["msa_mask_fraction"],
    }
    df_tm = pd.DataFrame(prt)
    merge_and_write(prev_df, df_tm, out_csv)
