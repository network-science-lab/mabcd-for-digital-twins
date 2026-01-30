from plot_utils import (
    load_multiple_divergence_scores,
    translate_fields,
    save_divergence_plot,
)

experiment_results_glob = (
    "./data/evaluate/experiment_3/bigreal/Freebase/exp_*/divergence_scores.csv"
)
df = load_multiple_divergence_scores(experiment_results_glob)
df["Estimation Method"] = df["results_path"].apply(lambda path: path.split("\\")[-2])
df.drop(columns=["results_path"], inplace=True)
df = df.melt(
    id_vars=["graph", "Estimation Method"],
    var_name="Divergence Metric",
    value_name="value",
)
df = translate_fields(df, metrics_colname="Divergence Metric", methods_colname="Estimation Method")
figfile = "./data/evaluate/experiment_3/bigreal/Freebase/experiment_3_divergence_by_method.pdf"
save_divergence_plot(df, figfile, hue="Estimation Method")
