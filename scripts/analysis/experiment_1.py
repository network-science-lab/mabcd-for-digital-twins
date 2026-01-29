import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

result_paths = glob(
    "./data/evaluate/experiment_1/bigreal/Freebase/d*/divergence_scores.csv"
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
    table["d"] = int(path.split("\\")[-2][1])
df = pd.concat(dfs, ignore_index=True)
df = df[~df.graph.isin(["Mean", "Std"])]
df.drop(columns=["mean_divergence"], inplace=True)
df = df.melt(
    id_vars=["graph", "d"],
    var_name="Divergence Metric",
    value_name="value",
)
rename_map = {
    "R_edges_correlation": "R",
    "tau_degrees_correlation": "τ",
    "r_communities_correlation": "r",
    "gamma_degree_distribution": "γ",
    "beta_community_sizes_distribution": "β",
    "xi_intercommunity_noise": "ξ",
}
df["Divergence Metric"] = df["Divergence Metric"].replace(rename_map)

# Plotting fancy
ax = sns.barplot(
    data=df,
    x="Divergence Metric",
    y="value",
    hue="d",
)
ax.set_yscale("log")
ax.set_ylim(0, 1)
plt.ylabel("Divergence Score")
plt.xlabel("Divergence Metric")
plt.savefig(
    "./data/evaluate/experiment_1/bigreal/Freebase/experiment_1_divergence_by_d.pdf"
)
plt.clf()
