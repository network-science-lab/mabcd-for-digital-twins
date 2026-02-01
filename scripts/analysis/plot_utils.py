from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

metrics_rename_map = {
    "R_edges_correlation": "R",
    "tau_degrees_correlation": "τ",
    "r_communities_correlation": "r",
    "gamma_degree_distribution": "γ",
    "beta_community_sizes_distribution": "β",
    "xi_intercommunity_noise": "ξ",
}

method_rename_map = {
    "exp_b": "Tuned r; loss r",
    "exp_e": "Tuned r+τ; loss r",
    "exp_f": "Tuned r+τ; loss τ",
    "exp_g": "Tuned r+τ; loss r+τ",
    "exp_i": "Tuned r+d; loss r",
}


def load_multiple_divergence_scores(path_glob: str, cut_means: bool = True) -> pd.DataFrame:
    result_paths = glob(path_glob)
    dfs = [
        pd.read_csv(
            path,
            header=0,
        )
        for path in result_paths
    ]
    for table, path in zip(dfs, result_paths):
        table["results_path"] = path
    df = pd.concat(dfs, ignore_index=True)
    if cut_means:
        df = df[~df.graph.isin(["Mean", "Std"])]
        df = df.drop(columns=["mean_divergence"])
    return df


def translate_fields(
    df: pd.DataFrame,
    metrics_colname: str | None = None,
    methods_colname: str | None = None,
) -> pd.DataFrame:
    if metrics_colname is not None:
        df[metrics_colname] = df[metrics_colname].replace(metrics_rename_map)
    if methods_colname is not None:
        df[methods_colname] = df[methods_colname].replace(method_rename_map)
    return df


def save_divergence_plot(df: pd.DataFrame, outfile: str, hue: str) -> None:
    ax = sns.barplot(
        data=df,
        x="Divergence Metric",
        y="value",
        hue=hue,
        palette="Blues",
        linewidth=0.5,
        edgecolor="black",
    )
    ax.set_yscale("log")
    ax.set_ylim(0, 1)
    plt.ylabel("Divergence Score")
    plt.xlabel("Divergence Metric")
    plt.savefig(outfile)
    plt.clf()
