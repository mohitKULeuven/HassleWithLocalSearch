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

plt.rcParams.update({"font.size": 20})

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--aggregate",
    nargs="*",
    type=str,
    default=[
        "model_learned",
        "score",
        "accuracy",
        "f1_score",
        "infeasiblity",
        "regret",
        "time_taken",
    ],
)
CLI.add_argument("--aggregate_over", nargs="*", type=str, default=["cutoff", "method"])
CLI.add_argument("--folder", type=str, default="results/06-05-20 (11:35:09.937930)/")
CLI.add_argument("--file", type=str, default="evaluation")
CLI.add_argument("--type", type=str, default="line")
args = CLI.parse_args()

result_file = args.folder + args.file + ".csv"

data = pd.read_csv(result_file)
data["score"][data["accuracy"] == -1] = np.nan
data["accuracy"] = data["accuracy"].replace(-1, np.nan)
data["f1_score"] = data["f1_score"].replace(-1, np.nan)
data["regret"] = data["regret"].replace(-1, np.nan)
data["accuracy"] = data["accuracy"] / 100
data["f1_score"] = data["f1_score"] / 100
data["infeasiblity"] = data["infeasiblity"] / 100
data["regret"] = data["regret"] / 100
data["score"] = data["score"] / 100


def std_err(x):
    return np.std(x) / np.sqrt(len(x))


if args.type == "line":
    fig, ax = plt.subplots(
        len(args.aggregate),
        3,
        figsize=(15, 3 * len(args.aggregate)),
        sharex="col",
        sharey="row",
    )
    for i, stats in enumerate(args.aggregate):
        for j, c in enumerate([25, 50, 100]):
            tmp_data = data.loc[data["num_context"] == c]
            tmp_data["model_learned"] = 1 - tmp_data["accuracy"].isna().astype(int)
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
            ax[i, j].get_legend().remove()
            ax[i, j].set_xticks([60, 300, 600, 900, 1200, 1500, 1800])
            ax[i, j].set_xticklabels([1, 5, 10, 15, 20, 25, 30])
            ax[i, j].set_ylabel(stats.capitalize())
            ax[i, j].set_xlabel("cutoff (in minutes)")
            ax[i, j].grid(True)
            ax[0, j].set_title(r"|$\mathcal{\Psi}$|=" + str(c))
            if stats == "model_learned":
                ax[i, j].set_ylim(0, 1.1)
            elif stats == "score":
                ax[i, j].set_ylim(0.8, 1.01)
            elif stats == "accuracy":
                ax[i, j].set_ylim(0.6, 0.9)
                ax[i, j].set_yticks([0.6, 0.7, 0.8, 0.9])
            elif stats == "f1_score":
                ax[i, j].set_ylim(0.3, 1)
            elif stats == "regret":
                ax[i, j].set_ylim(0, 0.03)
            elif stats == "infeasiblity":
                ax[i, j].set_ylim(0, 0.3)
            handles, labels = ax[i, j].get_legend_handles_labels()

    for i, l in enumerate(labels):
        if l == "adaptive_novelty_plus":
            labels[i] = "adaptive_novelty" + r"$^+$"
        if l == "novelty_plus":
            labels[i] = "novelty" + r"$^+$"
    lgd = fig.legend(handles=handles, labels=labels, loc="upper center", ncol=3)
    plt.savefig(
        args.folder + "synthetic_" + args.file + ".png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.2,
    )

elif args.type == "learned":
    fig, ax = plt.subplots(1, 3, figsize=(15, 3), sharex="col", sharey="row")
    for i, stats in enumerate(args.aggregate):
        for j, c in enumerate([25, 50, 100]):
            tmp_data = data.loc[data["num_context"] == c]
            tmp_data["model_learned"] = 1 - tmp_data["accuracy"].isna().astype(int)
            tmp_data["method"][tmp_data["method"] != "MILP"] = "SLS"

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
            line_mean_df.plot(rot=0, ax=ax[j], yerr=line_std_df)
            ax[j].get_legend().remove()
            ax[j].set_xticks([60, 300, 600, 900, 1200, 1500, 1800])
            ax[j].set_xticklabels([1, 5, 10, 15, 20, 25, 30])
            ax[j].set_ylabel(stats.capitalize())
            ax[j].set_xlabel("cutoff (in minutes)")
            ax[j].grid(True)
            ax[j].set_title(r"|$\mathcal{\Psi}$|=" + str(c))
            if stats == "model_learned":
                ax[j].set_ylim(-0.1, 1.1)
            handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.335, 1.3, 0.24, 0.1),
        mode="expand",
        ncol=2,
    )
    plt.savefig(
        args.folder + "synthetic_" + args.file + "_models_learned.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


elif args.type == "noise":
    fig, ax = plt.subplots(
        len(args.aggregate),
        4,
        figsize=(18, 3 * len(args.aggregate)),
        sharex="col",
        sharey="row",
    )
    for i, stats in enumerate(args.aggregate):
        for j, c in enumerate([10, 25, 50, 100]):
            tmp_data = data.loc[data["num_context"] == c]
            tmp_data["model_learned"] = 1 - tmp_data["accuracy"].isna().astype(int)

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
            ax[i, j].get_legend().remove()
            ax[i, j].set_xticks([60, 300, 600, 900, 1200, 1500, 1800])
            ax[i, j].set_xticklabels([1, 5, 10, 15, 20, 25, 30])
            ax[i, j].set_ylabel(stats.capitalize())
            ax[i, j].set_xlabel("cutoff (in minutes)")
            ax[i, j].grid(True)
            ax[0, j].set_title(r"|$\mathcal{\Psi}$|=" + str(c))
            if stats == "model_learned":
                ax[i, j].set_ylim(0, 1.1)
            elif stats == "score":
                ax[i, j].set_ylim(0.8, 1.01)
            elif stats == "accuracy":
                ax[i, j].set_ylim(0.6, 1)
            elif stats == "f1_score":
                ax[i, j].set_ylim(0.3, 1)
            elif stats == "regret":
                ax[i, j].set_ylim(0, 0.05)
            elif stats == "infeasiblity":
                ax[i, j].set_ylim(0, 0.3)
            handles, labels = ax[i, j].get_legend_handles_labels()
    fig.legend(
        handles=handles, labels=labels, loc="upper center", ncol=5, title="Methods"
    )
    plt.savefig(
        args.folder + "synthetic_" + args.file + ".png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
