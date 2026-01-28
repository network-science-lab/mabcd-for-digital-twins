import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
df = pd.read_csv(
    "./data/evaluate/experiment_finder_method/bigreal/Freebase/divergence_scores.csv",
    header=0,
)
df = df.iloc[: df.shape[0] - 2, :]
df.drop(columns=["mean_divergence"], inplace=True)
df["Estimation Method"] = df.graph.str.replace("Freebase-twin_edges_", "").str[:-2]
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
    "fancy_r": "Advanced r",
    "fancy_r_tau": "Advanced r+τ",
    "fancy_tau": "Advanced τ",
    "rudimentary": "Simple",
}
df["Divergence Metric"] = df["Divergence Metric"].replace(metric_rename_map)
df["Estimation Method"] = df["Estimation Method"].replace(method_rename_map)
# Plotting fancy
ax = sns.barplot(
    data=df,
    x="Divergence Metric",
    y="value",
    hue="Estimation Method",
)
ax.set_ylim(0, 1)
plt.ylabel("Divergence Score")
plt.xlabel("Divergence Metric")
plt.savefig(
    "./data/evaluate/experiment_finder_method/bigreal/Freebase/divergence_by_method.pdf"
)
plt.clf()
