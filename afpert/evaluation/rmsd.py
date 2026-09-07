import biotite.structure.io
import biotite.structure as struct
from biotite.structure import superimpose_homologs, superimpose
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import glob

from afpert.evaluation.incremental import load_done, merge_and_write

# def build_rmsd_db(structure1: str, structure2:str):

def build_rmsd_db_old(structure_1, structure_2, target_dir, method="afsample2"):

    # guides = [biotite.structure.io.load_structure("../6IRS_B.pdb"), biotite.structure.io.load_structure("../7DSQ_B.pdb")]
    guides = [biotite.structure.io.load_structure(f"../{structure_2}.pdb"), biotite.structure.io.load_structure(f"../{structure_1}.pdb")]

    guides = [struct.concatenate((guides[0][guides[0].atom_name == "CA"][:113], guides[0][guides[0].atom_name == "CA"][130:])), guides[1][guides[1].atom_name == "CA"][4:-1]]

    guides[1], _ = struct.superimpose(guides[0], guides[1])

    rms = struct.rmsd(guides[0], guides[1])

    print(np.around(rms, decimals=3))

    prots_rmsd = {
        "points": [[], []],
        "msa_batch": [],
        "mean_plddt": [],
        "subsample_depth": [],
        "msa_mask_fraction": [],
        "query_mask_fraction": [],
        "pert_seed": [],
    }

    all_pdbs = glob.glob(f"runs/{target_dir}/{method}/*/predictions/*/*_unrelaxed_rank_*.pdb")

    for i in tqdm(all_pdbs):

        try:
            mean_plddt = np.mean(json.load(open(i.replace("unrelaxed", "scores").replace(".pdb", ".json")))["plddt"])
        except Exception as e:
            print(f"warning, skipping {i}, got error {e}")
            continue

        mobile_struct = biotite.structure.io.load_structure(i)
        mobile_struct = struct.concatenate((mobile_struct[mobile_struct.atom_name == "CA"][46:159], mobile_struct[mobile_struct.atom_name == "CA"][176:488]))

        prots_rmsd["msa_batch"].append(0)
        prots_rmsd["subsample_depth"].append(int(i.split('predictions')[1].split("/")[1]))
        prots_rmsd["mean_plddt"].append(mean_plddt)
        prots_rmsd["query_mask_fraction"].append(int(i.split("/")[-1].split("_")[2])) # 2 for query mask
        prots_rmsd["msa_mask_fraction"].append(int(i.split("/")[-1].split("_")[3])) # 3 for msa mask
        prots_rmsd["pert_seed"].append(int(i.split("/")[-1].split("_")[4])) # 3 for msa mask

        for j in range(len(guides)):
            n = min(guides[j].array_length(), mobile_struct.array_length())
            idx = np.arange(n)
            fitted, _ = struct.superimpose(guides[j][idx], mobile_struct[idx])
            prots_rmsd["points"][j].append(np.around(struct.rmsd(guides[j][idx], fitted), decimals=3))
            
            
    prt_rmsd = {
    "7DSQ_rmsd": prots_rmsd["points"][1], 
    "6IRS_rmsd": prots_rmsd["points"][0],
    "msa_batch": prots_rmsd["msa_batch"],
    "mean_plddt": prots_rmsd["mean_plddt"],
    "MSA Samples": [int(x) for x in prots_rmsd["subsample_depth"]],
    "Query Mask %": prots_rmsd["query_mask_fraction"],
    "MSA Mask %": prots_rmsd["msa_mask_fraction"],
    "Perturbation Seed": prots_rmsd["pert_seed"]
    }
    df_rmsd = pd.DataFrame(prt_rmsd)
    df_rmsd.to_csv(f"runs/{target_dir}/{method}/rmsd_scores.csv")


def build_rmsd_db(structure_1, structure_2, target_dir, method="afsample2", incremental=False):
    out_csv = f"runs/{target_dir}/{method}/rmsd_scores.csv"
    prev_df, done = load_done(out_csv, incremental)

    ref2 = biotite.structure.io.load_structure(f"../{structure_2}.pdb")
    ref1 = biotite.structure.io.load_structure(f"../{structure_1}.pdb")
    # guides = [ref2[ref2.atom_name == "CA"], ref1[ref1.atom_name == "CA"][2:]]
    guides = [ref2[ref2.atom_name == "CA"], ref1[ref1.atom_name == "CA"]]

    # align by sequence and get rmsd for matched aas
    fitted1, _, g0_idx, g1_idx = superimpose_homologs(guides[0], guides[1])
    print(np.around(struct.rmsd(guides[0][g0_idx], fitted1[g1_idx]), decimals=3))

    prots_rmsd = {
        "points": [[], []],
        "path": [],
        "msa_batch": [],
        "mean_plddt": [],
        "subsample_depth": [],
        "msa_mask_fraction": [],
        "query_mask_fraction": [],
        "pert_seed": [],
    }

    all_pdbs = glob.glob(f"runs/{target_dir}/{method}/*/predictions/*/*_unrelaxed_rank_*.pdb")
    all_pdbs = [i for i in all_pdbs if i not in done]
    if incremental:
        print(f"{target_dir}/{method}: scoring {len(all_pdbs)} new prediction(s), "
              f"{len(done)} already in {out_csv}")

    for i in tqdm(all_pdbs):

        try:
            mean_plddt = np.mean(json.load(open(i.replace("unrelaxed", "scores").replace(".pdb", ".json")))["plddt"])
        except Exception as e:
            print(f"warning, skipping {i}, got error {e}")
            continue

        mobile_struct = biotite.structure.io.load_structure(i)
        # mobile_struct = struct.concatenate((mobile_struct[mobile_struct.atom_name == "CA"][46:159], mobile_struct[mobile_struct.atom_name == "CA"][176:488]))

        prots_rmsd["path"].append(i)
        prots_rmsd["msa_batch"].append(0)
        prots_rmsd["subsample_depth"].append(int(i.split('predictions')[1].split("/")[1]))
        prots_rmsd["mean_plddt"].append(mean_plddt)
        prots_rmsd["query_mask_fraction"].append(int(i.split("/")[-1].split("_")[1])) # 2 for query mask
        prots_rmsd["msa_mask_fraction"].append(int(i.split("/")[-1].split("_")[2])) # 3 for msa mask
        prots_rmsd["pert_seed"].append(int(i.split("/")[-1].split("_")[3])) # 3 for msa mask

        # mobile_struct_ca = mobile_struct[mobile_struct.atom_name == "CA"][2:]
        mobile_struct_ca = mobile_struct[mobile_struct.atom_name == "CA"]
        for guide_idx in range(len(guides)):
          fitted, _, gi, mi = superimpose_homologs(guides[guide_idx], mobile_struct_ca)
          # fitted, _ = superimpose(guides[guide_idx], mobile_struct_ca)
          prots_rmsd["points"][guide_idx].append(
              # np.around(struct.rmsd(guides[guide_idx][309:328], fitted[309:328]), decimals=3)
              np.around(struct.rmsd(guides[guide_idx][gi], fitted[mi]), decimals=3)
          )
          
          
    if '/' in structure_1:
        structure_1 = structure_1.split("/")[-1]
    
    if '/' in structure_2:
        structure_2 = structure_2.split("/")[-1]

            
            
    prt_rmsd = {
    f"{structure_1}_rmsd": prots_rmsd["points"][1],
    f"{structure_2}_rmsd": prots_rmsd["points"][0],
    "path": prots_rmsd["path"],
    "msa_batch": prots_rmsd["msa_batch"],
    "mean_plddt": prots_rmsd["mean_plddt"],
    "MSA Samples": [int(x) for x in prots_rmsd["subsample_depth"]],
    "Query Mask %": prots_rmsd["query_mask_fraction"],
    "MSA Mask %": prots_rmsd["msa_mask_fraction"],
    "Perturbation Seed": prots_rmsd["pert_seed"]
    }
    df_rmsd = pd.DataFrame(prt_rmsd)
    merge_and_write(prev_df, df_rmsd, out_csv)
