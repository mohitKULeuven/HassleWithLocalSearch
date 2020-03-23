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

plt.rcParams.update({'font.size': 15})

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--aggregate",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=[ "accuracy","f1_score","regret"]  # default if nothing is provided
)
CLI.add_argument(
    "--aggregate_over",  
    nargs="*",  
    type=str,
    default=["num_context","method"],  
)
CLI.add_argument(
    "--folder",  
    type=str,
    default="results/22-03-20 (14:05:00.078703)/",  
)
CLI.add_argument(
    "--file",  
    type=str,
    default="evaluation",  
)
CLI.add_argument(
    "--type",  
    type=str,
    default="line",  
)
#data_size=5
args = CLI.parse_args()

result_file = args.folder + args.file + ".csv"

data = pd.read_csv(result_file)
#data=data.loc[data["method"]=="adaptive_novelty_plus"]
data=data.loc[data["data_size"]==5]
#data=data.loc[data["cutoff"]==300]
data["accuracy"]=data["accuracy"]/100
data["f1_score"]=data["f1_score"]/100
data["regret"] = data["regret"].replace(-1,np.nan)
data["regret"]=data["regret"]/100
#data["f1_score"]=(2*data["recall"]*data["precision"])/(data["recall"]+data["precision"])
#print(data[["regret"]])
#data[["regret_c"]] = 1 - data[["regret_c"]]

def std_err(x):
    return np.std(x) / np.sqrt(len(x))



#mean_table = mean_table[args.aggregate]
if args.type=="line":
    fig, ax = plt.subplots(1, 3, figsize=(18,3))
#    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    
    for i,stats in enumerate(args.aggregate):
        mean_table = pd.pivot_table(
            data, [stats], index=args.aggregate_over, aggfunc=np.mean
        )
        std_table = pd.pivot_table(
            data, [stats], index=args.aggregate_over, aggfunc=std_err
        )
        
        mean_table_df = pd.DataFrame(mean_table.to_records())
        std_table_df = pd.DataFrame(std_table.to_records())
        line_mean_df=mean_table_df.pivot(index=args.aggregate_over[0],columns=args.aggregate_over[1], values=stats)
        line_std_df=std_table_df.pivot(index=args.aggregate_over[0],columns=args.aggregate_over[1], values=stats)
    #    print(line_mean_df)
        line_mean_df.plot(rot=0, ax=ax[i], yerr=line_std_df)
#        ax[i].legend(title=args.aggregate_over[1], fontsize="medium", ncol=1, loc='lower right',bbox_to_anchor=(1, -0.02))
        ax[i].get_legend().remove()
#        plt.ylim(0,100)
#        ax[i].set_color_cycle(['green','yellow','orange'])
#        ax[i].set_xticks([250,500,750,1000])
        ax[i].set_ylabel(stats)
        ax[i].grid(True)
        ax[i].set_xlabel(args.aggregate_over[0])
        handles, labels = ax[i].get_legend_handles_labels()
    plt.subplots_adjust(wspace=0.25)
    fig.legend(     # The line objects
           labels=labels,   # The labels for each line
           loc="center top",   # Position of legend
           bbox_to_anchor=(0.11, 1.02, 0.65, 0.2),
           mode="expand",
           ncol=4,
           borderaxespad=0.,    # Small spacing around legend box
           title="Methods"  # Title for the legend
           )
    plt.savefig(args.folder + "synthetic_"+ args.file + "_of_methods_over_num_context.png", 
                bbox_inches="tight", pad_inches=0)

else:
    mean_table = mean_table.T
    std_table = std_table.T
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    mean_table.plot.bar(rot=0, ax=ax, yerr=std_table, align="center", width=0.8)
    #ax.legend(title="$\hat{s}^+,s^-$", fontsize="large", ncol=1)
    ax.legend(title=args.aggregate_over, fontsize="medium", ncol=1, bbox_to_anchor=(0.7, 0.3))
    #plt.xticks(np.arange(4), ["precision", "regret"])
    plt.ylim(0,10)
    plt.savefig(args.folder + "synthetic_"+ args.file + "_over_"+args.aggregate_over[0]+".png", bbox_inches="tight", pad_inches=0)
plt.show()

