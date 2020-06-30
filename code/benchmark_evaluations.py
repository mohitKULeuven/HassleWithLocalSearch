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
from datetime import datetime
import json

from type_def import MaxSatModel
from generator import generate_contexts_and_data
from experiment import learn_model, learn_model_MILP, evaluate_statistics_sampling

logger = logging.getLogger(__name__)
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def cnf_to_model(cnf_file, num_constraints, seed):
    model = []
    n = 0
    k = 0
    w_hard = 0
    with open(cnf_file) as fp:
        line = fp.readline()
        p_ind = 0
        while line:
            line_as_list = line.strip().split()
            if len(line_as_list) > 1 and p_ind == 1 and represent_int(line_as_list[0]):
                if w_hard == 0:
                    model.append((None, set(map(int, line_as_list[:-1]))))
                else:
                    w = int(line_as_list[0])
                    if w == w_hard:
                        model.append((None, set(map(int, line_as_list[1:-1]))))
                    else:
                        model.append((w, set(map(int, line_as_list[1:-1]))))
            if p_ind == 0 and line_as_list[0] == "p":
                p_ind = 1
                n = int(line_as_list[2])
                k = int(line_as_list[3])
                if len(line_as_list) >= 5:
                    w_hard = int(line_as_list[4])
            line = fp.readline()
    if model:
        rng = np.random.RandomState(seed)
        if k > num_constraints:
            tmp_model = []
            indices = list(rng.choice(range(k), num_constraints, replace=False))
            for i in indices:
                tmp_model.append(model[i])
            model = tmp_model
            k = num_constraints
        return model, n, k
    return None


def cnf_param(cnf_file, num_constraints):
    with open(cnf_file) as fp:
        line = fp.readline()
        while line:
            line_as_list = line.strip().split()
            if line_as_list[0] == "p":
                if int(line_as_list[3]) > num_constraints:
                    return int(line_as_list[2]), num_constraints
                return int(line_as_list[2]), int(line_as_list[3])
            line = fp.readline()
    return 0, 0


def add_weights_cnf(model: MaxSatModel, k, num_soft, seed):
    num_hard = k - num_soft
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
            model, n, m = cnf_to_model(args.path + cnf_file, args.num_constraints, 111)
            for s in args.per_soft:
                for seed in args.model_seeds:
                    model = add_weights_cnf(model, m, int(m * s / 100), seed)
                    param = f"_{cnf_file}_num_constraints_{args.num_constraints}_per_soft_{s}_model_seed_{seed}"
                    pickle_var = {}
                    pickle_var["true_model"] = model
                    pickle.dump(
                        pickle_var,
                        open("pickles/target_model/" + param + ".pickle", "wb"),
                    )
                    for c, context_seed in it.product(
                        args.num_context, args.context_seeds
                    ):
                        tag = generate_contexts_and_data(
                            n, model, c, args.num_pos, args.num_neg, param, context_seed
                        )
                        print(tag)


def learn(args):
    for c, context_seed, method, t in it.product(
        args.num_context, args.context_seeds, args.method, args.cutoff
    ):
        if not args.filename:
            args.filename = os.listdir(args.path)
        for cnf_file in args.filename:
            if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                for s in args.per_soft:
                    for seed in args.model_seeds:
                        n, m = cnf_param(args.path + cnf_file, args.num_constraints)
                        param = f"_{cnf_file}_num_constraints_{args.num_constraints}_per_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                        if method=="MILP":
                            try:
                                learn_model_MILP(n, n, m, method, t, param)
                            except FileNotFoundError:
                                print("FileNotFound: " + param)
                                continue
                        else:
                            try:
                                learn_model(n, n, m, method, t, param, args.weighted)
                            except FileNotFoundError:
                                print("FileNotFound: " + param)
                                continue


def evaluate(args):
    folder_name = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
    os.mkdir(f"results/{folder_name}")
    with open(f"results/{folder_name}/arguments.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    csvfile = open(f"results/{folder_name}/evaluation.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "num_vars",
            "num_hard",
            "num_soft",
            "model_seed",
            "num_context",
            "context_seed",
            "num_pos",
            "num_neg",
            "pos_per_context",
            "neg_per_context",
            "method",
            "score",
            "recall",
            "precision",
            "accuracy",
            "f1_score",
            "regret",
            "infeasiblity",
            "f1_random",
            "regret_random", 
            "inf_random",
            "time_taken",
            "cutoff",
        ]
    )
    for s, seed in it.product(args.per_soft, args.model_seeds):
        if not args.filename:
            args.filename = os.listdir(args.path)
        for cnf_file in args.filename:
            if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                n, m = cnf_param(args.path + cnf_file, args.num_constraints)
                num_soft = int(m * s / 100)
                param = f"_{cnf_file}_num_constraints_{args.num_constraints}_per_soft_{s}_model_seed_{seed}"
                pickle_var = pickle.load(
                    open("pickles/target_model/" + param + ".pickle", "rb")
                )
                target_model = pickle_var["true_model"]
                max_t=max(args.cutoff)
                for c, context_seed, method, t in it.product(
                    args.num_context, args.context_seeds, args.method, args.cutoff
                ):
                    tag = (
                        param
                        + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}_method_{method}_cutoff_{max_t}"
                    )
                    if args.weighted == 0:
                        pickle_var = pickle.load(
                            open(
                                "pickles/learned_model/" + tag + ".pickle",
                                "rb",
                            )
                        )
                    else:
                        pickle_var = pickle.load(
                            open(
                                "pickles/learned_model/" + tag + ".pickle",
                                "rb",
                            )
                        )
                    i=0
                    if t<max_t:
                        for i,cutoff in enumerate(pickle_var["time_taken"]):
                            if cutoff>t:
                                break
                    learned_model = pickle_var["learned_model"][i-1]
                    time_taken = pickle_var["time_taken"][i-1]
                    score = pickle_var["score"][i-1]
                    contexts = pickle_var["contexts"]

                    global_context = set()
                    for context in contexts:
                        global_context.update(context)
                    recall, precision, accuracy, regret = -1, -1, -1, -1
                    if learned_model:
                        recall, precision, accuracy, regret, infeasiblity, f1_random, reg_random, inf_random= evaluate_statistics_sampling(
                            n,
                            target_model,
                            learned_model,
                            global_context,
                            args.sample_size,
                            seed,
                        )
                    f1_score = 0
                    if recall + precision != 0:
                        f1_score = 2 * recall * precision / (recall + precision)
                    if c == 0:
                        pos_per_context = pickle_var["labels"].count(True)
                        neg_per_context = pickle_var["labels"].count(False)
                    else:
                        pos_per_context = pickle_var["labels"].count(True) / c
                        neg_per_context = pickle_var["labels"].count(False) / c
                    print(
                        seed,
                        context_seed,
                        cnf_file,
                        m,
                        t,
                        score,
                        accuracy,
                        f1_score,
                        reg_random, regret, inf_random,
                        infeasiblity
                    )
                    filewriter.writerow(
                        [
                            n,
                            m - num_soft,
                            num_soft,
                            seed,
                            c,
                            context_seed,
                            args.num_pos,
                            args.num_neg,
                            pos_per_context,
                            neg_per_context,
                            method,
                            score,
                            recall,
                            precision,
                            accuracy,
                            f1_score,
                            regret,
                            infeasiblity, 
                            f1_random, reg_random, inf_random,
                            time_taken,
                            t,
                        ]
                    )
    csvfile.close()


logger = logging.getLogger(__name__)
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="generate")
    CLI.add_argument("--path", type=str, default="cnfs/3cnf_benchmark/")
    CLI.add_argument("--filename", nargs="*", type=str, default=[])
    CLI.add_argument("--per_soft", nargs="*", type=int, default=[10, 50])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_context", nargs="*", type=int, default=[25, 50])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_pos", type=int, default=10)
    CLI.add_argument("--num_neg", type=int, default=10)
    CLI.add_argument("--sample_size", type=int, default=1000)
    CLI.add_argument("--num_constraints", type=int, default=50)
    CLI.add_argument(
        "--method",
        nargs="*",
        type=str,
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
