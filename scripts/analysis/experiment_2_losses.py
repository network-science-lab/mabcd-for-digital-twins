from mfdt.config_finder.ff_loss import (
    tau_loss,
    r_loss,
    r_tau_loss,
)
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dfs = []
steps = list(range(1, 31))
for exp in ["exp_e", "exp_f", "exp_g"]:
    save_path = Path(f"data/finder/{exp}/bigreal/Freebase/logs")
    a = np.load(save_path / "A.npy")
    b = np.load(save_path / "B.npy")
    r = []
    r_best = []
    rb = np.inf

    tau = []
    tau_best = []
    tb = np.inf

    r_tau = []
    r_tau_best = []
    rtb = np.inf

    for i in steps:
        a_prime = np.load(save_path / str(i) / "A_primes.npy")
        b_prime = np.load(save_path / str(i) / "B_primes.npy")
        _r = r_loss(A=a, A_prime=a_prime)
        _tau = tau_loss(B=b, B_prime=b_prime)
        _r_tau = r_tau_loss(A=a, A_prime=a_prime, B=b, B_prime=b_prime)

        rb = rb if rb < _r else _r
        tb = tb if tb < _tau else _tau
        rtb = rtb if rtb < _r_tau else _r_tau

        r.append(_r)
        tau.append(_tau)
        r_tau.append(_r_tau)

        r_best.append(rb)
        tau_best.append(tb)
        r_tau_best.append(rtb)

    df = pd.DataFrame(
        {
            "r_loss": r,
            "tau_loss": tau,
            "r_tau_loss": r_tau,
            "r_loss_best": r_best,
            "tau_loss_best": tau_best,
            "r_tau_loss_best": r_tau_best,
            "iteration": steps,
        }
    )
    df["experiment"] = exp
    dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.melt(
    id_vars=["iteration", "experiment"], var_name="Loss", value_name="Loss Value"
)
merged_df["Loss Type"] = merged_df["Loss"].apply(
    lambda x: "best" if "best" in x else "current"
)
merged_df["Loss"] = merged_df["Loss"].replace(
    {
        "r_loss": "r",
        "tau_loss": "τ",
        "r_tau_loss": "r+τ loss",
        "r_loss_best": "r",
        "tau_loss_best": "τ",
        "r_tau_loss_best": "r+τ",
    }
)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
titles = ["Optimized for loss r", "Optimized for loss τ", "Optimized for loss r+τ"]
for i, exp in enumerate(["exp_e", "exp_f", "exp_g"]):
    df = merged_df[merged_df.experiment == exp]
    sns.lineplot(
        data=df[df["Loss Type"] == "best"],
        x="iteration",
        y="Loss Value",
        ax=ax[i],
        hue="Loss",
        linewidth=2.0,
        marker="p",
        markersize=6,
        palette="Blues",
    )
    ax[i].set_title(titles[i])
    ax[i].set_yscale("log")
    ax[i].set_ylim(0, 1)
    if i:
        ax[i].get_legend().remove()
        ax[i].set_ylabel("")
    if i == 1:
        ax[i].set_xlabel("Iteration")
    else:
        ax[i].set_xlabel("")
plt.savefig(
    "./data/evaluate/experiment_2/bigreal/Freebase/experiment_2_loss_best_by_method.pdf"
)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
titles = ["Optimized for loss r", "Optimized for loss τ", "Optimized for loss r+τ"]
for i, exp in enumerate(["exp_e", "exp_f", "exp_g"]):
    df = merged_df[merged_df.experiment == exp]
    sns.lineplot(
        data=df[df["Loss Type"] == "current"],
        x="iteration",
        y="Loss Value",
        ax=ax[i],
        hue="Loss",
        linewidth=2.0,
        linestyle="--",
        # marker="p",
        # markersize=6,
        palette="Blues",
    )
    ax[i].set_title(titles[i])
    ax[i].set_yscale("log")
    ax[i].set_ylim(0, 1)
    if i:
        ax[i].get_legend().remove()
        ax[i].set_ylabel("")
    if i == 1:
        ax[i].set_xlabel("Iteration")
    else:
        ax[i].set_xlabel("")
plt.savefig(
    "./data/evaluate/experiment_2/bigreal/Freebase/experiment_2_loss_current_by_method.pdf"
)
