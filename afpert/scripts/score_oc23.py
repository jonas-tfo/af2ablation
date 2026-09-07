
import argparse
import csv
import glob
import multiprocessing as mp
import os
from pathlib import Path

import biotite.structure as struct
import biotite.structure.io
import pandas as pd
from biotite.sequence import ProteinSequence
from tmtools import tm_align
from tqdm import tqdm

OC23_ROOT = "afsample_paper_models/generated_models/oc23"
METHODS = ["afsample", "afvanilla", "msasubsampling", "afsample2", "SPEACH_AF"]
FLAT_METHODS = {"afsample", "afvanilla", "msasubsampling"}
OUT_NAME = "final_df_tmalign.csv"


def three2one(ca) -> str:
    out = []
    for res_name in ca.res_name:
        try:
            out.append(ProteinSequence.convert_letter_3to1(res_name))
        except KeyError:
            out.append("X")
    return "".join(out)


def load_ca(path: str):
    arr = biotite.structure.io.load_structure(path)
    ca = arr[arr.atom_name == "CA"]
    return ca.coord, three2one(ca)


def read_ref_map(oc23_csv: str) -> dict[str, tuple[str, str]]:
    m: dict[str, tuple[str, str]] = {}
    with open(oc23_csv) as f:
        for r in csv.DictReader(f):
            m[r["Uniprotid"]] = (r["pdbid_open"], r["pdbid_closed"])
    return m


def _unit(out_csv: Path, target: str, ref_map, pdbs: list[str]) -> dict:
    o, c = ref_map[target]
    return {
        "out_csv": str(out_csv),
        "target": target,
        "pdbid_o": o,
        "pdbid_c": c,
        "open_pdb": f"pdbs/open/{o}.pdb",
        "closed_pdb": f"pdbs/closed/{c}.pdb",
        "pdbs": pdbs,
    }


def build_units(methods, targets, ref_map) -> list[dict]:
    root = Path(OC23_ROOT)
    units: list[dict] = []
    for method in methods:
        if method in FLAT_METHODS:  # <method>/<target>/unrelaxed_*.pdb
            for t in targets:
                d = root / method / t
                pdbs = sorted(glob.glob(str(d / "unrelaxed_*.pdb")))
                if pdbs:
                    units.append(_unit(d / OUT_NAME, t, ref_map, pdbs))
        elif method == "afsample2":  # <method>/<ablation>/<target>/unrelaxed_*.pdb
            mroot = root / method
            for abl in sorted(os.listdir(mroot)):
                for t in targets:
                    d = mroot / abl / t
                    pdbs = sorted(glob.glob(str(d / "unrelaxed_*.pdb")))
                    if pdbs:
                        units.append(_unit(d / OUT_NAME, t, ref_map, pdbs))
        elif method == "SPEACH_AF":  # pool <target>_<id> dirs into one per bare target
            base = root / method / "speachafout_oc23"
            by_t: dict[str, list[str]] = {}
            for sub in sorted(os.listdir(base)):
                t = sub.rsplit("_", 1)[0]
                if t not in targets:
                    continue
                by_t.setdefault(t, []).extend(
                    sorted(glob.glob(str(base / sub / "unrelaxed_*.pdb")))
                )
            for t, pdbs in by_t.items():
                if pdbs:
                    units.append(_unit(root / method / t / OUT_NAME, t, ref_map, pdbs))
    return units


def score_unit(u: dict) -> tuple[str, int, str]:
    try:
        oc, os_ = load_ca(u["open_pdb"])
        cc, cs = load_ca(u["closed_pdb"])
    except Exception as e:
        return (u["out_csv"], 0, f"ref load failed: {e}")

    rows, fail = [], 0
    for p in u["pdbs"]:
        try:
            mc, ms = load_ca(p)
            tm_open = float(tm_align(mc, oc, ms, os_).tm_norm_chain2)
            tm_close = float(tm_align(mc, cc, ms, cs).tm_norm_chain2)
        except Exception:
            fail += 1
            continue
        rows.append({
            "model_pdb": os.path.basename(p),
            "pdbid_o": u["pdbid_o"],
            "pdbid_c": u["pdbid_c"],
            "TM_open": tm_open,
            "TM_close": tm_close,
        })

    if rows:
        Path(u["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(u["out_csv"], index=False)
    return (u["out_csv"], len(rows), f"{fail} pdb(s) failed" if fail else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", action="append", choices=METHODS, help="restrict to this method (repeatable); default: all")
    ap.add_argument("--jobs", type=int, default=min(20, os.cpu_count() or 1))
    ap.add_argument("--oc23-csv", default="oc23.csv")
    ap.add_argument("--dry-run", action="store_true", help="print unit/pdb counts and exit without scoring")
    args = ap.parse_args()

    methods = args.method or METHODS
    ref_map = read_ref_map(args.oc23_csv)
    targets = set(ref_map)

    units = build_units(methods, targets, ref_map)
    keep, skipped = [], []
    for u in units:
        if os.path.exists(u["open_pdb"]) and os.path.exists(u["closed_pdb"]):
            keep.append(u)
        else:
            skipped.append(u)

    n_pdb = sum(len(u["pdbs"]) for u in keep)
    print(f"{len(keep)} unit(s), {n_pdb} pdb(s) to score across methods={methods}")
    if skipped:
        miss = sorted({u["target"] for u in skipped})
        print(f"{len(skipped)} unit(s) skipped (missing ref pdb) for targets: {miss}")
    if args.dry_run:
        return

    with mp.Pool(args.jobs) as pool:
        for out_csv, n, msg in tqdm(
            pool.imap_unordered(score_unit, keep), total=len(keep)
        ):
            if msg:
                tqdm.write(f"{out_csv}: {n} rows, {msg}")
    print("done")


if __name__ == "__main__":
    main()
