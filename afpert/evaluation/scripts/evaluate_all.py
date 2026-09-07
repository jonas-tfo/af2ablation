#!/usr/bin/env python3

from afpert.evaluation.tm import build_tm_db
from afpert.evaluation.rmsd import build_rmsd_db
from pathlib import Path
import argparse
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--target-dir", type=str, help="dir to save the csv to")
    parser.add_argument("--structure1", type=str, help="struct 1")
    parser.add_argument("--structure2", type=str, help="struct 2")
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom"], default="afsample2", help="perturbation method subtree to evaluate")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.tm:
        build_tm_db(args.structure1, args.structure2, args.target_dir, method=args.method)
    if args.rmsd:
        build_rmsd_db(args.structure1, args.structure2, args.target_dir, method=args.method)
















# PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# print(PROJECT_ROOT)


# guides = [
#     load_guide(PROJECT_ROOT / "6IRS_B.pdb"),
#     load_guide(PROJECT_ROOT / "7DSQ_B.pdb"),
# ]

# # runs/<target>/<target>_<q>_<m>_<pertseed>/predictions/<depth>/*_unrelaxed_*.pdb
# PDB_RX = re.compile(r"_unrelaxed_rank_(\d+)_alphafold2_model_(\d+)_seed_(\d+)\.pdb$")

# pdbs = sorted(Path("runs/7DSQ_2").glob("*/predictions/*/*_unrelaxed_*.pdb"))

# rows = []
# for pdb in tqdm(pdbs):
#     m = PDB_RX.search(pdb.name)
#     if not m:
#         continue
#     rank, model, seed = map(int, m.groups())
#     run_dir = pdb.parents[2].name            # targrt subdir <target>_<q>_<m>_<pertseed>
#     depth = int(pdb.parents[0].name)       # just folder name
#     _, q, mm, pert_seed = run_dir.rsplit("_", 3)         # name, qmask, mmask, perturbation seed
#     q, mm, pert_seed = int(q), int(mm), int(pert_seed)

#     tm = tm_against_guides(pdb, guides)

#     # get plddt from json next to pdb
#     score_path = pdb.with_name(pdb.name.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json"))
#     plddt = None
#     if score_path.exists():
#         with score_path.open() as f:
#             scores = json.load(f)
#         plddt = sum(scores["plddt"]) / len(scores["plddt"])

#     rows.append({
#         "q": q, "m": mm, "depth": depth, "pert_seed": pert_seed,
#         "model": model, "seed": seed, "rank": rank,
#         "tm_if": tm["6IRS_B"],
#         "tm_of": tm["7DSQ_B"],
#         "plddt": plddt,
#     })

# df = pd.DataFrame(rows)
# df.to_csv("runs/7DSQ_2/tm_scores.csv", index=False) 
