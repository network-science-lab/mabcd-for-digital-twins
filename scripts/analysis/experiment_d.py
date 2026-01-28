import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
df = pd.read_csv(
    "./data/evaluate/experiment_d/bigreal/Freebase/divergence_scores.csv",
    header=0,
)
df = df.iloc[: df.shape[0] - 2, :]
df.drop(columns=["mean_divergence"], inplace=True)
df["d"] = df["graph"].str[-3:-2]
df["estimation_method"] = df.graph.str.contains("fancy").map(
    {True: "fancy_r", False: "rudimentary"}
)
df = df.melt(
    id_vars=["graph", "d", "estimation_method"],
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
    data=df[df.estimation_method == "fancy_r"],
    x="Divergence Metric",
    y="value",
    hue="d",
)
ax.set_ylim(0, 1)
plt.title("Advanced configuration retrieval")
plt.ylabel("Divergence Score")
plt.xlabel("Divergence Metric")
plt.savefig("./data/evaluate/experiment_d/bigreal/Freebase/divergenceby_d_fancy_r.pdf")
plt.clf()

# Plotting rudimentary

ax = sns.barplot(
    data=df[df.estimation_method == "rudimentary"],
    x="Divergence Metric",
    y="value",
    hue="d",
)
ax.set_ylim(0, 1)
plt.title("Simple configuration retrieval")
plt.ylabel("Divergence Score")
plt.xlabel("Divergence Metric")
plt.savefig(
    "./data/evaluate/experiment_d/bigreal/Freebase/divergence_by_d_rudimentary.pdf"
)
plt.clf()
