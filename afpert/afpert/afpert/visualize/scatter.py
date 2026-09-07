from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_scatter(df: pd.DataFrame, x: str, y: str, *, title: str, save_path: Path, **kwargs,):
    sns.set_theme()
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, ax=ax, **kwargs)
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)
    return fig
