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

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--aggregate",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=[ "precision","recall","regret",]  # default if nothing is provided
)
CLI.add_argument(
    "--aggregate_over",  
    nargs="*",  
    type=str,
    default=["num_context"],  
)
CLI.add_argument(
    "--folder",  
    type=str,
    default="results/26-02-20 (14:33:20.511289)",  
)
CLI.add_argument(
    "--file",  
    type=str,
    default="evaluation",  
)

args = CLI.parse_args()

result_file = args.folder + "/" + args.file + ".csv"

data = pd.read_csv(result_file)
data[["regret"]] = data[["regret"]].replace(-1,np.nan)
print(data[["regret"]])
#data[["regret_c"]] = 1 - data[["regret_c"]]

def std_err(x):
    return np.std(x) / np.sqrt(len(x))

mean_table = pd.pivot_table(
    data, args.aggregate, index=args.aggregate_over, aggfunc=np.mean
)
std_table = pd.pivot_table(
    data, args.aggregate, index=args.aggregate_over, aggfunc=std_err
)
mean_table = mean_table[args.aggregate]
print(mean_table)

mean_table = mean_table.T

std_table = std_table.T

print(mean_table)
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
mean_table.plot.bar(rot=0, ax=ax, yerr=std_table, align="center", width=0.8)
ax.legend(title="$\hat{s}^+,s^-$", fontsize="large", ncol=1)
#plt.xticks(
#    np.arange(4), ["precision", "regret"]
#)
#plt.savefig("plots/" + args.file + ".png", bbox_inches="tight", pad_inches=0)
plt.show()

