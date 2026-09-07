#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from afpert.visualize import benchmark, benchmarkdumbell, rmsd, rmsf, tm, bestopenclosed, boxplots

from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
import pandas as pd

# afpert/scripts/visualize.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Plot perturbation scores.")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--rmsf", action="store_true", help="plot per-target RMSF (pooled, per-combo, cross-method)")
    parser.add_argument("--rmsf-aggregate", action="store_true", help="cross-target RMSF effect boxplot over the whole dataset")
    parser.add_argument("--benchmark", action="store_true", help="plot all per-combination benchmark heatmaps")
    parser.add_argument("--dumbell", action="store_true", help="plot dumbbell overview")
    parser.add_argument("--panels", action="store_true", help="best open vs best closed panels across all methods")
    parser.add_argument("--boxplot", action="store_true", help="")
    parser.add_argument("--dataset", required=True, help="dataset csv file")
    parser.add_argument("--rmsf-summary", action="store_true", help="")
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom", "afsample", "afvanilla", "msasubsampling"], default="afsample2", help="perturbation method subtree to plot")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)


    df = pd.read_csv(args.dataset)
    pdbparser = PDBParser()
    io = PDBIO()

    for _, row in df.iterrows():
        if not os.path.exists("./pdbs/open/"+row["pdbid_open"]+'.pdb') or not os.path.exists("./pdbs/closed/"+row["pdbid_closed"]+'.pdb'):

            structures = [pdbparser.get_structure(row["pdbid_open"].split("_")[0], "./pdbs/open/"+row["pdbid_open"].split("_")[0]+".pdb"), pdbparser.get_structure(row["pdbid_closed"].split("_")[0], "./pdbs/closed/"+row["pdbid_closed"].split("_")[0]+".pdb")]
            for structure in range(len(structures)):
                pdb_chains = structures[structure].get_chains()
                prefix = ""
                if structure == 0:
                    prefix = "pdbs/open/"
                else:
                    prefix = "pdbs/closed/"
                for chain in pdb_chains:
                    io.set_structure(chain)
                    io.save(f"{prefix}{structures[structure].get_id()}_{chain.get_id()}.pdb")

        try:
            if args.tm:
                tm.plot_all(args.method, row['Uniprotid'], row['pdbid_open'], row['pdbid_closed'])
            if args.rmsd:
                rmsd.plot_all(args.method, row['Uniprotid'], row['pdbid_open'], row['pdbid_closed'])
            if args.rmsf:
                rmsf.plot_all(args.method, row['Uniprotid'])
                rmsf.compare_per_residue_rmsf(row['Uniprotid'])
            if args.benchmark:
                benchmark.plot_all(args.method, row['Uniprotid'])
            if args.dumbell:
                benchmarkdumbell.plot_all(args.method, row['Uniprotid'],row['pdbid_open'], row['pdbid_closed'])
            if args.panels:
                bestopenclosed.panels(row['Uniprotid'], row['pdbid_open'], row['pdbid_closed'])
            if args.boxplot:
                boxplots.plot()

            if getattr(args, "rmsf_aggregate", False):
                rmsf.aggregate(targets=df["Uniprotid"].tolist(), methods=[args.method])

            if getattr(args, "rmsf_summary", False):
                rmsf.summary_over_targets(targets=df["Uniprotid"].tolist(), methods=[args.method])

        except Exception as e:
            print(e)

    if getattr(args, "rmsf_aggregate", False):
        rmsf.aggregate(targets=df["Uniprotid"].tolist(), methods=[args.method])


if __name__ == "__main__":
    main()
