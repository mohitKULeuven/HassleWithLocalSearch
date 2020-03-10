#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 08:58:30 2019

@author: mohit
"""


import numpy as np
import logging
import pickle
import csv
import argparse
import itertools as it
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

from type_def import MaxSatModel
from generator import generate_contexts_and_data
from experiment import learn_model, evaluate_statistics
logger = logging.getLogger(__name__)
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def cnf_to_model(cnf_file):
    model = []
    n = 0
    k = 0
    w_hard=0
    with open(cnf_file) as fp:
        line = fp.readline()
        p_ind = 0
        while line:
            line_as_list = line.strip().split()
            if len(line_as_list) > 1 and p_ind == 1 and represent_int(line_as_list[0]):
                if w_hard==0:
                    model.append((None, set(map(int, line_as_list[:-1]))))
                else:
                    w=int(line_as_list[0])
                    if w==w_hard:
                        model.append((None, set(map(int, line_as_list[1:-1]))))
                    else:
                        model.append((w, set(map(int, line_as_list[1:-1]))))
            if p_ind == 0 and line_as_list[0] == "p":
                p_ind = 1
                n = int(line_as_list[2])
                k = int(line_as_list[3])
                if len(line_as_list)>=5:
                    w_hard=int(line_as_list[4])
            line = fp.readline()
    if model:
        return model, n, k
    return None

def cnf_param(cnf_file):
    with open(cnf_file) as fp:
        line = fp.readline()
        while line:
            line_as_list = line.strip().split()
            if line_as_list[0] == "p":
                return int(line_as_list[2]),int(line_as_list[3])
            line = fp.readline()
    return 0,0


def add_weights_cnf(model: MaxSatModel, k, num_soft, seed):
    num_hard=k-num_soft
    rng = np.random.RandomState(seed)
    indices = list(range(k))
    hard_indices = list(sorted(rng.permutation(indices)[:num_hard]))
    soft_indices = list(sorted(set(indices) - set(hard_indices)))

    weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=len(soft_indices))

    for i, weight in zip(soft_indices, weights):
        model[i] = (weight, model[i][1])
    return model


def generate(args):
    for cnf_file in os.listdir(args.path):
        if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
            model, n, m = cnf_to_model(args.path + cnf_file)
            for s in args.per_soft:
                for seed in args.model_seeds:
                    model=add_weights_cnf(model, m, int(m*s/100), seed)
                    param = f"_{cnf_file}_per_soft_{s}_model_seed_{seed}"
                    for c, context_seed, d in it.product(
                        args.num_context, args.context_seeds, args.data_size
                    ):
                        tag = generate_contexts_and_data(n, model, c, d, param, context_seed)
                        print(tag)

def learn(args):
    for c, context_seed, d, method, t in it.product(
        args.num_context,args.context_seeds,
        args.data_size,args.method,args.cutoff,
    ):
        for cnf_file in os.listdir(args.path):
            if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                for s in args.per_soft:
                    for seed in args.model_seeds:
                        n, m = cnf_param(args.path + cnf_file)
                        param = f"_{cnf_file}_per_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}"
                        try:
                            learn_model(n, n, m, method, t,param,args.weighted)
                        except FileNotFoundError:
                            print(param)
        #                    exit
                            continue

def evaluate(args):
    folder_name=datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
    os.mkdir(f"results/{folder_name}")
#    os.mkdir(f"results/{folder_name}/weighted/")
    with open(f"results/{folder_name}/arguments.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    csvfile = open(f"results/{folder_name}/evaluation.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "num_vars","num_hard","num_soft","model_seed",
            "num_context","context_seed","data_size",
            "pos_per_context","neg_per_context","method",
            "score","recall","precision","regret","time_taken","cutoff"
        ]
    )
    for s, seed in it.product(args.per_soft, args.model_seeds):
        for cnf_file in os.listdir(args.path):
            if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                n, m = cnf_param(args.path + cnf_file)
                num_soft = int(m * s/100)
                param = f"_{cnf_file}_per_soft_{s}_model_seed_{seed}"
                target_model, n, m = cnf_to_model(args.path + cnf_file)
                for c, context_seed, d, method, t in it.product(
                    args.num_context,args.context_seeds,args.data_size,
                    args.method,args.cutoff,
                ):
                    tag = (param
                            + f"_num_context_{c}_num_data_{d}_context_seed_{context_seed}_method_{method}_cutoff_{t}"
                            )
                    if args.weighted==0:
                        pickle_var = pickle.load(
                            open("pickles/bin_weight/learned_model" + tag + ".pickle", "rb")
                        )
                    else:
                        pickle_var = pickle.load(
                            open("pickles/con_weight/learned_model" + tag + ".pickle", "rb")
                        )
                    learned_model = pickle_var["learned_model"]
                    time_taken = pickle_var["time_taken"]
                    score = pickle_var["score"]
                    recall, precision, regret = evaluate_statistics(
                        n, target_model, learned_model, args.sample_size
                    )
                    pos_per_context=pickle_var["labels"].count(True)/c
                    neg_per_context=pickle_var["labels"].count(False)/c
                    print(n, m-num_soft, num_soft, d, score, recall, 
                          precision, regret,time_taken)
                    filewriter.writerow(
                            [n,m-num_soft,num_soft,seed,c,context_seed,d,
                             pos_per_context,neg_per_context,method,
                             score,recall,precision,regret,time_taken,t]
                    )
    csvfile.close()

def avg_training_score(args):
    for method in args.method:
        for t in args.cutoff:
            score = []
            for s, seed, c, context_seed, d in it.product(
                args.per_soft,args.model_seeds,args.num_context,
                args.context_seeds,args.data_size
            ):
                for cnf_file in os.listdir(args.path):
                    if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                        try:
                            tag = f"_{cnf_file}_per_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}_method_{method}_cutoff_{t}"
                            if args.weighted==0:
                                pickle_var = pickle.load(
                                    open("pickles/bin_weight/learned_model" + tag + ".pickle", "rb")
                                )
                            else:
                                pickle_var = pickle.load(
                                    open("pickles/con_weight/learned_model" + tag + ".pickle", "rb")
                                )
                            score.append(pickle_var["score"])
                        except FileNotFoundError:
                            continue
            if score:
                avg=np.mean(score)
                print(f"method:{method} time:{t} score:{avg}")

def save_training_score_plot(args):
    fig,ax=plt.subplots()
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Accuracy (Training Data)')
    plt.xlabel('cutoff time (in seconds)')
    plt.ylim(0,100)
    for method in args.method:
        avg_score = []
        for s, seed, c, context_seed, d in it.product(
            args.per_soft,args.model_seeds,args.num_context,
            args.context_seeds,args.data_size
        ):
            for cnf_file in os.listdir(args.path):
                if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                    score = []
                    for t in args.cutoff:
                        try:
                            tag = f"_{cnf_file}_per_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}_method_{method}_cutoff_{t}"
                            if args.weighted==0:
                                pickle_var = pickle.load(
                                    open("pickles/bin_weight/learned_model" + tag + ".pickle", "rb")
                                )
                            else:
                                pickle_var = pickle.load(
                                    open("pickles/con_weight/learned_model" + tag + ".pickle", "rb")
                                )
                            score.append(pickle_var["score"])
                        except FileNotFoundError:
                            continue
                    if score:
                        avg_score.append(score)
        avg_score=np.array(avg_score)
        y=np.average(avg_score,axis=0)
#        y_err=[elem/np.sqrt(len(avg_score[i])) for i,elem in enumerate(np.std(avg_score,axis=0))]
        print(method,y)
        plt.plot(args.cutoff, np.average(avg_score,axis=0), label=method)
#        plt.fill_between(args.cutoff, y-y_err, y+y_err,alpha=0.3)
        plt.legend(loc="upper left")
        plt.draw()
    tag = f"results/benchmark_evaluation_over_score.png"
    fig.savefig(tag)
            

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="generate")
    CLI.add_argument("--path", type=str, default="cnfs/3cnf_benchmark/")
    CLI.add_argument("--per_soft", nargs="*", type=int, default=[10, 50])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_context", nargs="*", type=int, default=[25,50])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--data_size", nargs="*", type=int, default=[5, 25, 50])
    CLI.add_argument("--sample_size", type=int, default=1000)
    CLI.add_argument("--method",nargs="*",type=str,
        default=["walk_sat", "novelty", "novelty_plus", "adaptive_novelty_plus"],
    )
    CLI.add_argument("--cutoff", nargs="*", type=int, default=[2, 10, 60])
    CLI.add_argument("--weighted", type=int, default=1)
    args = CLI.parse_args()
    
    if args.function == "generate":
        generate(args)

    elif args.function == "learn":
        learn(args)

    elif args.function == "evaluate":
        evaluate(args)
        
    elif args.function == "print_score":
        avg_training_score(args)
        
    elif args.function == "plot_score":
        save_training_score_plot(args)
        
        
        
        
        
        
        
        
        
        
        
        