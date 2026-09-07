#!/usr/bin/env python3

from afpert.evaluation.tm import build_tm_db_tmalign, build_tm_db
from afpert.evaluation.rmsd import build_rmsd_db
from afpert.evaluation.rmsf import build_rmsf_db
import concurrent.futures
from pathlib import Path
import argparse
import os
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics")
    parser.add_argument("--tm", action="store_true", help="plot all TM-score figures")
    parser.add_argument("--rmsd", action="store_true", help="plot all RMSD figures")
    parser.add_argument("--rmsf", action="store_true", help="build per-residue RMSF (pooled + per-combo)")
    parser.add_argument("--method", choices=["afsample2", "alphamask", "custom"], default="afsample2", help="perturbation method subtree to evaluate")
    parser.add_argument("--tmmethod", choices=["tmalign", "biotite"], default="tmalign", help="tm score backend")
    parser.add_argument("--dataset", type=str, help="dir to save the csv to")
    parser.add_argument("--incremental", action="store_true", help="only score predictions not already in the existing tm/rmsd csv")
    args = parser.parse_args()
    
    build_tm = build_tm_db_tmalign if args.tmmethod == "tmalign" else build_tm_db

    os.chdir(PROJECT_ROOT)
    df = pd.read_csv(args.dataset)
    pdbparser = PDBParser()
    io = PDBIO()

    rows = [row for _, row in df.iterrows()]

    print(rows)

    def process(row): 
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

        if args.tm:
            try:
                build_tm("localcolabfold/pdbs/open/"+row["pdbid_open"], "localcolabfold/pdbs/closed/"+row["pdbid_closed"], row["Uniprotid"], method=args.method, incremental=args.incremental)
            except Exception as e:
                print(e)
        if args.rmsd:
            build_rmsd_db("localcolabfold/pdbs/open/"+row["pdbid_open"], "localcolabfold/pdbs/closed/"+row["pdbid_closed"], row["Uniprotid"], method=args.method, incremental=args.incremental)
        if args.rmsf:
            try:
                build_rmsf_db("localcolabfold/pdbs/open/"+row["pdbid_open"], "localcolabfold/pdbs/closed/"+row["pdbid_closed"], row["Uniprotid"], method=args.method)
            except Exception as e:
                print(e)

    for row in rows: 
        process(row)
    
            