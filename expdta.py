import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import glob
    from biopandas.pdb import PandasPdb
    from prody import parsePDBHeader
    from typing import Optional
    from pathlib import Path
    from typing import Tuple
    from collections import Counter
    from tqdm import tqdm

    def read_pdb_to_dataframe(
        pdb_path: Optional[str] = None,
        model_index: int = 1,
        parse_header: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates 
        and the Experimental Data (EXPDTA) string.

        Args:
            pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
            model_index (int, optional): Index of the model to extract. Defaults to 1.
            parse_header (bool, optional): Whether to parse the PDB header for EXPDTA. 
                Defaults to True.

        Returns:
            Tuple[pd.DataFrame, Optional[str]]: A tuple containing:
                - The DataFrame of atomic coordinates (ATOM and HETATM).
                - The EXPDTA string (or None if not found/parsed).
        """
        # 1. Extract Atomic Coordinates
        ppdb = PandasPdb().read_pdb(pdb_path)
        atomic_df = ppdb.get_model(model_index)
    
        if len(atomic_df.df["ATOM"]) == 0:
            raise ValueError(f"No model found for index: {model_index}")
    
        # Combine ATOM and HETATM records
        combined_df = pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])

        # 2. Extract EXPDTA from Header
        expdta = None
        if parse_header:
            # parsePDBHeader returns a dictionary of header records
            header = parsePDBHeader(pdb_path)
            # .get() is safer than bracket notation to avoid KeyError if EXPDTA is missing
            expdta = header.get('experiment')

        return combined_df, expdta

    # df, experiment_type = read_pdb_to_dataframe("protein.pdb")
    # print(f"Experimental Method: {experiment_type}")
    return Counter, glob, read_pdb_to_dataframe, tqdm


@app.cell
def _(read_pdb_to_dataframe):
    # df, df_header = read_pdb_to_dataframe("/home/friedrich/localcolabfold/pdbs/open/2rqm.pdb")
    # df.head(10)
    df, experiment_type = read_pdb_to_dataframe("/home/friedrich/localcolabfold/pdbs/open/2rqm.pdb")
    print(f"Experimental Method: {experiment_type}")
    return


@app.cell
def _(Counter, glob, tqdm):
    def get_expdta(pdb_path):
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("EXPDTA"):
                    return line[6:].strip()
                # header ends once coordinates start
                if line.startswith(("ATOM", "HETATM", "MODEL")):
                    break
        return None

    counts = Counter()
    methods = {}

    for path in tqdm(glob.glob("/home/friedrich/localcolabfold/pdbs/**/*.pdb", recursive=True)):
        expdta = get_expdta(path)
        methods[path] = expdta
        if expdta and "NMR" in expdta.upper():
            counts["NMR"] += 1
        elif expdta and "X-RAY" in expdta.upper():
            counts["X-RAY"] += 1
        else:
            counts["other"] += 1

    print(counts)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
