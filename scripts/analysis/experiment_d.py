import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
df = pd.read_csv(
    "./experiments/experiment_d/bigreal/Freebase/evaluate/divergence_scores.csv",
    header=0,
)
df = df.iloc[: df.shape[0] - 2, :]
df.drop(columns=["mean_divergence"], inplace=True)
df["d"] = df["index"].str[-3:-2]
df = df.melt(id_vars=["index", "d"], var_name="Divergence Metric", value_name="value")
ax = sns.lineplot(data=df, x="d", y="value", hue="Divergence Metric", marker="o")
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel("Divergence Score")
plt.xlabel("d")
plt.show()
