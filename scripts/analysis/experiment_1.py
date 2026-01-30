from plot_utils import load_multiple_divergence_scores, translate_fields, save_divergence_plot

experiment_results_glob = (
    "./data/evaluate/experiment_1/bigreal/Freebase/d*/divergence_scores.csv"
)
df = load_multiple_divergence_scores(experiment_results_glob)
df["d"] = df["results_path"].apply(lambda path: path.split("\\")[-2][1:])
df.drop(columns=["results_path"], inplace=True)
df = df.melt(
    id_vars=["graph", "d"],
    var_name="Divergence Metric",
    value_name="value",
)
df = translate_fields(df, metrics_colname="Divergence Metric")
figfile = "./data/evaluate/experiment_1/bigreal/Freebase/experiment_1_divergence_by_d.pdf"
save_divergence_plot(df, figfile, hue="d")