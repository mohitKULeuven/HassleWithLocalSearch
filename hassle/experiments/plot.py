import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from mpl_toolkits import mplot3d
import os
import glob

plt.rcParams.update({"font.size": 20})

def std_err(x):
    return np.std(x) / np.sqrt(len(x))


########### for MILP vs SLS ##############
def MILPvsSLS(data):
    tmp_data = data.loc[data["cutoff"] >= 600]
    # print(tmp_data["method"])
    tmp_data = tmp_data.loc[(tmp_data["method"] == "walk_sat") | (tmp_data["method"] == "MILP")]
    # print(tmp_data["method"])
    milp_data = tmp_data.loc[(tmp_data["score"] > 0) & (tmp_data["method"] == "MILP")]
    milp_data = milp_data[["model_seed", "context_seed", "max_cutoff"]]
    tmp_data = pd.merge(tmp_data, milp_data, on=["model_seed", "context_seed", "max_cutoff"])
    mean_table = pd.pivot_table(
        tmp_data, args.aggregate, index=["method"], aggfunc=np.mean
    )
    line_mean_df = pd.DataFrame(mean_table.to_records())
    print(line_mean_df)
    std_table = pd.pivot_table(
        tmp_data, args.aggregate, index=["method"], aggfunc=std_err
    )
    line_std_df = pd.DataFrame(std_table.to_records())
    print(line_std_df)
##########################################


########### for noisy data ##############
def Noisy(args, data):
    # tmp_data = data.loc[data["cutoff"] >= 600]
    # print(tmp_data["method"])
    # tmp_data = tmp_data.loc[(tmp_data["method"] == "walk_sat") | (tmp_data["method"] == "MILP")]
    # # print(tmp_data["method"])
    # milp_data = tmp_data.loc[(tmp_data["score"] > 0) & (tmp_data["method"] == "MILP")]
    # milp_data = milp_data[["model_seed", "context_seed", "max_cutoff"]]
    # tmp_data = pd.merge(tmp_data, milp_data, on=["model_seed", "context_seed", "max_cutoff"])
    mean_table = pd.pivot_table(
        data, args.aggregate, index=["neg_type"], aggfunc=np.mean
    )
    line_mean_df = pd.DataFrame(mean_table.to_records())
    print(line_mean_df)
    std_table = pd.pivot_table(
        data, args.aggregate, index=["neg_type"], aggfunc=std_err
    )
    line_std_df = pd.DataFrame(std_table.to_records())
    print(line_std_df)
##########################################



linestyles = ["s-", "o-", "^-", "D-", ">-"]

########### comparing all SLS methods ##############
def compareSLS_synthetic(args, data):
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
            tmp_data = tmp_data.loc[tmp_data["cutoff"] >= 600]
            tmp_data = tmp_data.loc[(tmp_data["method"] != "MILP")]

            mean_table = pd.pivot_table(
                tmp_data, [stats], index=args.aggregate_over, aggfunc=np.mean
            )
            line_mean_df = pd.DataFrame(mean_table.to_records()).pivot(
                index=args.aggregate_over[0],
                columns=args.aggregate_over[1],
                values=stats,
            )

            # std_table = pd.pivot_table(
            #     tmp_data, [stats], index=args.aggregate_over, aggfunc=std_err
            # )
            # line_std_df = pd.DataFrame(std_table.to_records()).pivot(
            #     index=args.aggregate_over[0],
            #     columns=args.aggregate_over[1],
            #     values=stats,
            # )

            mins = [6, 60, 600, 1200, 1800, 2400, 3000, 3600]
            line_mean_df.plot(rot=0, ax=ax[i, j], style=linestyles)
            ax[i, j].get_legend().remove()
            ax[i, j].set_xticks(mins)
            ax[i, j].set_xticklabels([int(x / 60) for x in mins])
            ax[i, j].set_ylabel(stats.capitalize())
            ax[i, j].set_xlabel("cutoff (in minutes)")
            ax[i, j].grid(True)
            ax[0, j].set_title(r"|$\mathcal{\Psi}$|=" + str(c))
            # if stats == "model_learned":
            #     ax[i, j].set_ylim(0, 1.1)
            # elif stats == "score":
            #     ax[i, j].set_ylim(0.85, 1.01)
            # elif stats == "accuracy":
            #     ax[i, j].set_ylim(0.65, 0.9)
            #     ax[i, j].set_yticks([0.6, 0.7, 0.8, 0.9])
            # elif stats == "f1_score":
            #     ax[i, j].set_ylim(0.3, 1)
            # if stats == "regret":
            #     ax[i, j].set_ylim(0.05, 0.15)
            #     ax[i, j].set_yticks([0.06, 0.1, 0.14])
            # elif stats == "infeasiblity":
                # ax[i, j].set_ylim(0, 0.25)
            handles, labels = ax[i, j].get_legend_handles_labels()
    for i, l in enumerate(labels):
        if l == "adaptive_novelty_plus":
            labels[i] = "adaptive_novelty" + r"$^+$"
        if l == "novelty_plus":
            labels[i] = "novelty" + r"$^+$"
        if l=="0":
            labels[i] = "context not used"
        if l=="1":
            labels[i] = "context used"

    lgd = fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.35, 1.0, 0.2, 0),
    )
    plt.savefig(
        args.folder + "synthetic_" + args.file + ".png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.savefig(
        args.folder + "synthetic_" + args.file + ".pdf",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.35,
    )

def compareModelsLearned(args, data):
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.figure(figsize=(5, 3))
    tmp_data=data.copy()
    tmp_data["model_learned"] = 1 - tmp_data["accuracy"].isna().astype(int)
    tmp_data["method"][tmp_data["method"] != "MILP"] = "SLS"

    # if tmp_data["model_learned"][tmp_data["method"] == "MILP"].isnull().all():
    #     tmp_data["model_learned"][tmp_data["method"] == "MILP"] = -1
    mean_table = pd.pivot_table(tmp_data, ["model_learned"], index=["num_context", "method"], aggfunc=np.mean)
    std_table = pd.pivot_table(tmp_data, ["model_learned"], index=["num_context", "method"], aggfunc=std_err)
    mean_table_df = pd.DataFrame(mean_table.to_records())
    std_table_df = pd.DataFrame(std_table.to_records())
    line_mean_df = mean_table_df.pivot(index="num_context", columns="method", values="model_learned")
    line_std_df = std_table_df.pivot(index="num_context", columns="method", values="model_learned")
    line_mean_df.plot(rot=0, style=linestyles, yerr=line_std_df)
    # ax[i].get_legend().remove()
    plt.xticks([25, 50, 100])
    # ax[i].set_xlabel("Number of Contexts")
    plt.ylabel("Models Learned")
    # ax[i].grid(True)
    # ax.set_title(r"|$\mathcal{\Psi}$|=" + str(c))
    # if stats == "model_learned":
    #     ax[i].set_ylim(-0.1, 1.1)
    # handles, labels = ax[i].get_legend_handles_labels()
    # lgd = fig.legend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=(0.4, 1.15, 0.2, 0.1),
    #     loc="upper center",
    #     ncol=2,
    # )
    plt.savefig(
        args.folder + "synthetic_" + args.file + "_models_learned.pdf",
        # args.folder + "models_learned_vs_num_context.png",
        # bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.savefig(
        args.folder + "synthetic_" + args.file + "_models_learned.png",
        # args.folder + "models_learned_vs_num_context.png",
        # bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.02,
    )


def noisyData(args, data):
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


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--aggregate",
        nargs="*",
        type=str,
        default=[
            # "model_learned",
            "score",
            "accuracy",
            "infeasiblity",
            "regret",
        ],
    )
    # CLI.add_argument("--aggregate_over", nargs="*", type=str, default=["cutoff", "method"])
    CLI.add_argument("--folder", type=str, default="results/neg_type/")
    CLI.add_argument("--file", type=str, default="evaluation")
    args = CLI.parse_args()

    all_csv_files = [
        file
        for path, subdir, files in os.walk(args.folder)
        for file in glob.glob(path + "/*.csv")
    ]
    data = pd.concat((pd.read_csv(f) for f in all_csv_files))

    # result_file = args.folder + args.file + ".csv"
    # data = pd.read_csv(result_file)
    # data["score"][data["accuracy"] == -1] = np.nan
    data["score"] = data["score"].replace(-1, np.nan)
    data["accuracy"] = data["accuracy"].replace(-1, np.nan)
    data["infeasiblity"] = data["infeasiblity"].replace(-1, np.nan)
    data["regret"] = data["regret"].replace(-1, np.nan)
    data["accuracy"] = data["accuracy"] / 100
    data["infeasiblity"] = data["infeasiblity"] / 100
    data["regret"] = data["regret"] / 100
    data["score"] = data["score"] / 100

    Noisy(args, data)
    # compareModelsLearned(args, data)
    # MILPvsSLS(data)