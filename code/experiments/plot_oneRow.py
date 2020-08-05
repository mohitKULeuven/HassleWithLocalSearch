#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:56:00 2020

@author: mohit
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

plt.rcParams.update({"font.size": 15})

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--aggregate",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=["score", "accuracy", "infeasiblity", "regret"],
)
CLI.add_argument(
    "--aggregate_over", nargs="*", type=str, default=["num_soft", "num_hard"]
)
CLI.add_argument(
    "--folder", nargs="*", type=str, default=["results/num_soft/", "results/num_hard/"]
)
CLI.add_argument("--file", type=str, default="evaluation")
args = CLI.parse_args()


def std_err(x):
    return np.std(x) / np.sqrt(len(x))


fig, ax = plt.subplots(
    2, len(args.aggregate_over), figsize=(12, 3.5), sharex="col", sharey="row"
)
colors = ["blue", "orange", "red", "green"]
for i, var in enumerate(args.aggregate_over):
    result_file = args.folder[i] + args.file + ".csv"
    data = pd.read_csv(result_file)

    data["score"][data["accuracy"] == -1] = np.nan
    data["accuracy"] = data["accuracy"].replace(-1, np.nan)
    data["infeasiblity"] = data["infeasiblity"].replace(-1, np.nan)
    data["regret"] = data["regret"].replace(-1, np.nan)
    #    data["regret"] = data["infeasiblity"].replace(-1, np.nan)
    data["accuracy"] = data["accuracy"] / 100
    data["f1_score"] = data["f1_score"] / 100
    data["infeasiblity"] = data["infeasiblity"] / 100
    data["regret"] = data["regret"] / 100
    data["score"] = data["score"] / 100

    h, l = [], []
    for j, stats in enumerate(args.aggregate):
        r = int(j / 2)
        tmp_data = data.loc[data["method"] == "walk_sat"]
        # tmp_data = data
        tmp_data["model_learned"] = 1 - tmp_data["accuracy"].isna().astype(int)

        #        tmp_data["infeasible"] = tmp_data["regret"].isna()

        if tmp_data[stats][tmp_data["method"] == "MILP"].isnull().all():
            tmp_data[stats][tmp_data["method"] == "MILP"] = -1
        mean_table = pd.pivot_table(tmp_data, [stats], index=[var], aggfunc=np.mean)
        std_table = pd.pivot_table(tmp_data, [stats], index=[var], aggfunc=std_err)
        mean_table.plot(rot=0, ax=ax[r, i], yerr=std_table, color=colors[j])
        ax[r, i].get_legend().remove()
        ax[r, 0].set_xticks([2, 5, 10, 15])
        ax[r, 1].set_xticks([2, 5, 10, 15])
        ax[r, i].grid(True)
        if j % 2 == 1:
            handles, labels = ax[r, i].get_legend_handles_labels()
            h.extend(handles)
            l.extend(labels)
fig.legend(  # The line objects
    handles=h,
    labels=l,
    loc="upper center",  # Position of legend
    bbox_to_anchor=(0.1, 1.02, 0.69, 0.1),
    #            mode="expand",
    ncol=4,
    #            borderaxespad=0.1,    # Small spacing around legend box
    #            title="Statistics"  # Title for the legend
)
plt.savefig(
    args.folder[0]
    + "synthetic_"
    + args.file
    + "_over_"
    + "_".join(args.aggregate_over)
    + ".png",
    bbox_inches="tight",
    pad_inches=0.1,
)
