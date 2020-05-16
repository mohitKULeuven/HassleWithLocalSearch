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
    default=[
        "score",
        "accuracy",
        "infeasiblity",
        "inf_random",
        "regret",
        "regret_random",
#        "time_taken"
    ],  # default if nothing is provided
)
CLI.add_argument(
    "--aggregate_over", nargs="*", type=str, default=["num_context"]
)
CLI.add_argument(
    "--folder",
    type=str,
    default="results/10-05-20 (15:45:42.127329)/"
)
CLI.add_argument("--file", type=str, default="evaluation")
args = CLI.parse_args()


def std_err(x):
    return np.std(x) / np.sqrt(len(x))


fig, ax = plt.subplots(1, 1, figsize=(12, 7), sharex="col", sharey="row")
for f, var in enumerate(args.aggregate_over):
    result_file = args.folder + args.file + ".csv"
    data = pd.read_csv(result_file)
    
    data["score"][data["accuracy"] == -1] = np.nan
    data["accuracy"] = data["accuracy"].replace(-1, np.nan)
    data["f1_score"] = data["f1_score"].replace(-1, np.nan)
    data["regret"] = data["regret"].replace(-1, np.nan)
    data["regret_random"] = data["regret_random"].replace(-1, np.nan)
    data["infeasiblity"] = data["infeasiblity"].replace(-1, np.nan)
    data["inf_random"] = data["inf_random"].replace(-1, np.nan)
#    data["regret"] = data["infeasiblity"].replace(-1, np.nan)
    data["accuracy"] = data["accuracy"] / 100
    data["f1_score"] = data["f1_score"] / 100
    data["infeasiblity"] = data["infeasiblity"] / 100
    data["regret"] = data["regret"] / 100
    data["inf_random"] = data["inf_random"] / 100
    data["regret_random"] = data["regret_random"] / 100
    data["score"] = data["score"] / 100
    
    data=data.loc[data["context_seed"]==111]
    

    for i, stats in enumerate(args.aggregate):
        mean_table = pd.pivot_table(data, [stats], index=[var], aggfunc=np.mean)
        std_table = pd.pivot_table(data, [stats], index=[var], aggfunc=std_err)
        print(mean_table)
        mean_table.plot(rot=0, ax=ax, yerr=std_table)
#        ax.get_legend().remove()
#        ax[0].set_xticks([8, 10, 12, 15])
#        ax[1].set_xticks([2, 5, 10, 15])
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
#fig.legend(  # The line objects
#    handles=handles,
#    labels=labels,
#    loc="upper center",  # Position of legend
#    bbox_to_anchor=(0.1, 1.02, 0.69, 0.1),
#    #            mode="expand",
#    ncol=5,
#    #            borderaxespad=0.1,    # Small spacing around legend box
#    #            title="Statistics"  # Title for the legend
#)
#plt.savefig(
#    args.folder[0]
#    + "synthetic_"
#    + args.file
#    + "_over_"
#    + "_".join(args.aggregate_over)
#    + ".png",
#    bbox_inches="tight",
#    pad_inches=0.1,
#)
