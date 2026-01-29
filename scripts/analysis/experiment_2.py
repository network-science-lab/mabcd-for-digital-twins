import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

result_paths = glob(
    "./data/evaluate/experiment_2/bigreal/Freebase/exp_*/divergence_scores.csv"
)
# Data preparation
dfs = [
    pd.read_csv(
        path,
        header=0,
    )
    for path in result_paths
]
for table, path in zip(dfs, result_paths):
    table["Estimation Method"] = path.split("\\")[-2]

df = pd.concat(dfs, ignore_index=True)
df = df[~df.graph.isin(["Mean", "Std"])]
df.drop(columns=["mean_divergence"], inplace=True)
df = df.melt(
    id_vars=["graph", "Estimation Method"],
    var_name="Divergence Metric",
    value_name="value",
)
metric_rename_map = {
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
}
df["Divergence Metric"] = df["Divergence Metric"].replace(metric_rename_map)
df["Estimation Method"] = df["Estimation Method"].replace(method_rename_map)

ax = sns.barplot(
    data=df,
    x="Divergence Metric",
    y="value",
    hue="Estimation Method",
)
ax.set_yscale("log")
ax.set_ylim(0, 1)
plt.ylabel("Divergence Score")
plt.xlabel("Divergence Metric")
plt.savefig(
    "./data/evaluate/experiment_2/bigreal/Freebase/experiment_2_divergence_by_method.pdf"
)
plt.clf()
