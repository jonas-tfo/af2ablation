import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns

    return pd, sns


@app.function
def get_pareto_front(df, x_col, y_col):
    """
    Returns the subset of the dataframe that forms the Pareto Front
    (maximizing both x and y).
    """

    df_sorted = df.sort_values(by=[x_col, y_col], ascending=False).reset_index(drop=False)
    
    pareto_front = []
    max_y = -float('inf')
    
    for idx, row in df_sorted.iterrows():
        if row[y_col] > max_y:
            pareto_front.append(row['index'])
            max_y = row[y_col]
            
    return df.loc[pareto_front]


@app.cell
def _(pd):
    structure = "P00558"
    df = pd.read_csv(f"runs/{structure}/custom/tm_scores.csv")
    afsampledf = pd.read_csv(f"afsample_paper_models/generated_models/oc23/afsample2/af_io_abl_15/{structure}/final_df_tmalign.csv")
    #afsampledf = pd.read_csv("afsample_paper_models/generated_models/oc23/msasubsampling/P00558/final_df_tmalign.csv")
    cols = []
    for x in df.columns:
        if x.endswith("_tm"): cols.append(x)
    
    pareto = get_pareto_front(df, cols[0], cols[1])
    afsamplepareto = get_pareto_front(afsampledf, "TM_close","TM_open")
    afsamplepareto
    return afsamplepareto, cols, pareto


@app.cell
def _(afsamplepareto, cols, pareto, sns):
    sns.scatterplot(pareto, x=cols[0], y=cols[1])
    sns.scatterplot(afsamplepareto, x="TM_open",y="TM_close")
    return


@app.cell
def _(pareto):
    for j in pareto["path"].tolist():
        print(j)
    return


@app.cell
def _(afsamplepareto):
    import zipfile
    from pathlib import Path

    with zipfile.ZipFile('pareto_P00558_subsample.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file, I1, I2  in zip(afsamplepareto.model_pdb, afsamplepareto["TM_open"], afsamplepareto["TM_close"]):
            # Convert to Path object for easier manipulation
            path = "afsample_paper_models/generated_models/oc23/msasubsampling/P00558" / Path(file)
        
            if path.exists():
                # arcname=path.name stores only the filename (e.g., 'data.csv')
                # instead of the full system path.
                zipf.write(path, arcname=str(I1)+"_"+str(I2)+".pdb")
                print(f"Added: {path.name}")
            else:
                print(f"Warning: File not found, skipping: {file}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
