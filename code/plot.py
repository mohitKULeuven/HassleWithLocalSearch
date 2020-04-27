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
from mpl_toolkits import mplot3d

plt.rcParams.update({"font.size": 15})

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--aggregate",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=[
        "model_learned",
        "score",
        "accuracy",
        "f1_score",
        "infeasible",
        "regret",
        "time_taken",
    ],  # default if nothing is provided
)
CLI.add_argument("--aggregate_over", nargs="*", type=str, default=["cutoff", "method"])
CLI.add_argument("--folder", type=str, default="results/26-04-20 (22:34:30.254998)/")
CLI.add_argument("--file", type=str, default="evaluation")
CLI.add_argument("--type", type=str, default="line")
# data_size=5
args = CLI.parse_args()

result_file = args.folder + args.file + ".csv"

data = pd.read_csv(result_file)
# milp_data=data.loc[data["method"]=="MILP"]
# print(milp_data[milp_data["accuracy"]==-1].shape[0])
# data=data.loc[data["method"]=="adaptive_novelty_plus"]
data = data.loc[data["cutoff"] != 50]
# data=data.loc[data["cutoff"]==1200]
# data=data.loc[data["num_context"]==10]
data["score"][data["accuracy"] == -1] = np.nan
data["accuracy"] = data["accuracy"].replace(-1, np.nan)
data["f1_score"] = data["f1_score"].replace(-1, np.nan)
data["regret"] = data["regret"].replace(-1, np.nan)
data["accuracy"] = data["accuracy"] / 100
data["f1_score"] = data["f1_score"] / 100
data["regret"] = data["regret"] / 100
data["score"] = data["score"] / 100
# data["f1_score"]=(2*data["recall"]*data["precision"])/(data["recall"]+data["precision"])
# print(data[["regret"]])
# data[["regret_c"]] = 1 - data[["regret_c"]]


def std_err(x):
    return np.std(x) / np.sqrt(len(x))


# mean_table = mean_table[args.aggregate]
if args.type == "line":
    fig, ax = plt.subplots(
        len(args.aggregate),
        5,
        figsize=(18, 3 * len(args.aggregate)),
        sharex="col",
        sharey="row",
    )
    #    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for i, stats in enumerate(args.aggregate):
        for j, c in enumerate([10, 25, 50, 100, 150]):
            tmp_data = data.loc[data["num_context"] == c]
            tmp_data["model_learned"] = 1 - tmp_data["accuracy"].isna().astype(int)
            tmp_data["infeasible"] = tmp_data["regret"].isna()

            #        tmp_data['optimal_not_feasible']=tmp_data["regret"].isna()
            #        print(tmp_data['regret'].tail())
            #        print(tmp_data["regret"].isna().sum()*100/tmp_data.shape[0])

            if tmp_data[stats][tmp_data["method"] == "MILP"].isnull().all():
                tmp_data[stats][tmp_data["method"] == "MILP"] = -1
            mean_table = pd.pivot_table(
                tmp_data, [stats], index=args.aggregate_over, aggfunc=np.mean
            )
            std_table = pd.pivot_table(
                tmp_data, [stats], index=args.aggregate_over, aggfunc=std_err
            )
            mean_table_df = pd.DataFrame(mean_table.to_records())
            std_table_df = pd.DataFrame(std_table.to_records())
            line_mean_df = mean_table_df.pivot(
                index=args.aggregate_over[0],
                columns=args.aggregate_over[1],
                values=stats,
            )
            line_std_df = std_table_df.pivot(
                index=args.aggregate_over[0],
                columns=args.aggregate_over[1],
                values=stats,
            )
            line_mean_df.plot(rot=0, ax=ax[i, j], yerr=line_std_df)
            #        ax[i,j].legend(title=args.aggregate_over[1], fontsize="medium", ncol=1, loc='lower right',bbox_to_anchor=(1, -0.02))
            ax[i, j].get_legend().remove()
            ax[i, j].set_xticks([60, 300, 600, 900, 1200, 1500, 1800])
            ax[i, j].set_xticklabels([1, 5, 10, 15, 20, 25, 30])
            ax[i, j].set_ylabel(stats.capitalize())
            ax[i, j].set_xlabel("cutoff (in minutes)")
            ax[i, j].grid(True)
            ax[-1, j].set_yticks([60, 300, 600, 900, 1200, 1500, 1800])
            ax[-1, j].set_yticklabels([1, 5, 10, 15, 20, 25, 30])
            ax[0, j].set_title(r"|$\mathcal{\Psi}$|=" + str(c))
            if stats == "model_learned":
                ax[i, j].set_ylim(0, 1.1)
            elif stats == "score":
                ax[i, j].set_ylim(0.7, 1.1)
            elif stats == "accuracy":
                ax[i, j].set_ylim(0.6, 1)
            elif stats == "f1_score":
                ax[i, j].set_ylim(0.3, 1)
            elif stats == "regret":
                ax[i, j].set_ylim(0, 0.2)
            handles, labels = ax[i, j].get_legend_handles_labels()
    #            print(labels)
    #    plt.subplots_adjust(wspace=0.25)
    fig.legend(  # The line objects
        handles=handles,
        labels=labels,
        loc="upper center",  # Position of legend
        #            bbox_to_anchor=(0., 1.02, 1, 0.1),
        #            mode="expand",
        ncol=5,
        #            borderaxespad=0.1,    # Small spacing around legend box
        title="Methods",  # Title for the legend
    )
    plt.savefig(
        args.folder + "synthetic_" + args.file + ".png",
        bbox_inches="tight",
        pad_inches=0,
    )

else:
    ax = plt.axes(projection="3d")
    data = data.loc[data["method"] == "adaptive_novelty_plus"]
    #    if tmp_data[stats][tmp_data["method"]=='MILP'].isnull().all():
    #        tmp_data[stats][tmp_data["method"]=='MILP']=-1
    mean_table = pd.pivot_table(
        data, ["precision"], index=args.aggregate_over, aggfunc=np.mean
    )
    std_table = pd.pivot_table(
        data, ["precision"], index=args.aggregate_over, aggfunc=std_err
    )
    mean_table_df = pd.DataFrame(mean_table.to_records())
    #    print(mean_table_df.columns)
    x, y = np.meshgrid(mean_table_df["cutoff"], mean_table_df["num_context"])
    #    print(x)
    z = mean_table.values
    ##    mean_table_df = pd.DataFrame(mean_table.to_records())
    ax.plot_wireframe(x, y, z, color="r")
#    std_table_df = pd.DataFrame(std_table.to_records())
#    line_mean_df=mean_table_df.pivot(index=args.aggregate_over[0],columns=args.aggregate_over[1], values=stats)
#    line_std_df=std_table_df.pivot(index=args.aggregate_over[0],columns=args.aggregate_over[1], values=stats)
#    line_mean_df.plot(rot=0, ax=ax[i,j], yerr=line_std_df)

#    plt.savefig(args.folder + "synthetic_"+ args.file +".png",
#                bbox_inches="tight", pad_inches=0)
plt.show()
