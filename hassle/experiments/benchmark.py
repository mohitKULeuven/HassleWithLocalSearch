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

from hassle.type_def import MaxSatModel, Context
from hassle.generator import generate_contexts_and_data
from hassle.experiments.synthetic import (
    learn_model_sls,
    learn_model_MILP,
    regret,
    evaluate_statistics,
    get_learned_model,
)
from hassle.pysat_solver import solve_weighted_max_sat, get_value, label_instance
from hassle.verify import get_recall_precision_sampling
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


def generate(cnf_file, h, s, seed, nc, num_pos, num_neg, neg_type, c_seed):
    path = "cnfs/3cnf_benchmark/"
    adaptive_seed = seed
    tag = False
    while not tag:
        if adaptive_seed-seed>100:
            break
        model, n, m = cnf_to_model(path + cnf_file, h+s, adaptive_seed)
        model = add_weights_cnf(model, m, s, adaptive_seed)
        param = f"_{cnf_file}_num_hard_{h}_num_soft_{s}_model_seed_{adaptive_seed}"
        pickle_var = {}
        pickle_var["true_model"] = model
        if not os.path.exists("pickles/target_model"):
            os.makedirs("pickles/target_model")
        pickle.dump(
            pickle_var,
            open("pickles/target_model/" + param + ".pickle", "wb"),
        )
        tag = generate_contexts_and_data(
            n,
            model,
            nc,
            num_pos,
            num_neg,
            neg_type,
            param,
            c_seed,
        )
        adaptive_seed += 1
    tqdm.write(tag)


def learn(cnf_file, h, s, seed, c, num_pos, num_neg, neg_type, context_seed, method, t, p):
    path = "cnfs/3cnf_benchmark/"
    adaptive_seed=seed
    found = False
    while not found:
        if adaptive_seed-seed>100:
            break
        n, m = cnf_param(path + cnf_file, h+s)
        param = f"_{cnf_file}_num_hard_{h}_num_soft_{s}_model_seed_{adaptive_seed}_num_context_{c}_num_pos_{num_pos}_num_neg_{num_neg}_neg_type_{neg_type}_context_seed_{context_seed}"
        if os.path.exists("pickles/contexts_and_data/" + param + ".pickle"):
            found = True
        adaptive_seed += 1
    if method == "MILP":
        try:
            learn_model_MILP(m, method, t, param, p, 1)
        except FileNotFoundError:
            print("FileNotFound: " + param)
    else:
        try:
            learn_model_sls(m, method, t, param, p, 1)
        except FileNotFoundError:
            print("FileNotFound: " + param)


def evaluate(cnf_file, h, s, seed, c, num_pos, num_neg, neg_type, context_seed, m, t, p):
    path="cnfs/3cnf_benchmark/"
    adaptive_seed = seed
    n, _ = cnf_param(path + cnf_file, h+s)
    max_t = 86400
    found = False
    while not found:
        if adaptive_seed-seed>100:
            break
        param = f"_{cnf_file}_num_hard_{h}_num_soft_{s}_model_seed_{adaptive_seed}"
        tag_cnd = (
                param
                + f"_num_context_{c}_num_pos_{num_pos}_num_neg_{num_neg}_context_seed_{context_seed}"
        )
        if neg_type:
            tag_cnd = (
                    param
                    + f"_num_context_{c}_num_pos_{num_pos}_num_neg_{num_neg}_neg_type_{neg_type}_context_seed_{context_seed}"
            )
        if not os.path.exists("pickles/contexts_and_data/" + tag_cnd + ".pickle"):
            adaptive_seed += 1
            continue
        found = True
        target_model = pickle.load(
            open("pickles/target_model/" + param + ".pickle", "rb")
        )["true_model"]
        pickle_cnd = pickle.load(
            open("pickles/contexts_and_data/" + tag_cnd + ".pickle", "rb")
        )

    if p == 0:
        p = int(p)
    tag = tag_cnd + f"_method_{m}_cutoff_{max_t}_noise_{p}"
    # if args.naive == 1:
    #     tag += "_naive"
    # if bl == 1:
    #     tag += "_bl"
    pickle_var = pickle.load(
        open("pickles/learned_model/" + tag + ".pickle", "rb")
    )
    if c == 0:
        c = 1
    labels = [True if l == 1 else False for l in pickle_cnd["labels"]]
    pos_per_context = labels.count(True) / c
    neg_per_context = labels.count(False) / c
    recall, precision, accuracy = (-1, -1, -1)
    regret, infeasiblity, f1_score = (-1, -1, -1)
    # print("time taken: ", pickle_var["time_taken"])
    index = get_learned_model(pickle_var["time_taken"], max_t, t)
    # print(t, index)
    time_taken = t
    iteration = 0
    num_nbr = 0
    score = -1
    if index is not None:
        learned_model = pickle_var["learned_model"][index]
        time_taken = pickle_var["time_taken"][index]
        if m != "MILP":
            iteration = pickle_var["iterations"][index]
            num_nbr = pickle_var["num_neighbour"][index]
        if learned_model:
            score = pickle_var["score"][index]

        # contexts = pickle_cnd["contexts"]
        # global_context = set()
        # for context in contexts:
        #     global_context.update(context)
        global_context = None
        if learned_model:
            (
                recall,
                precision,
                accuracy,
                regret,
                infeasiblity,
            ) = evaluate_statistics_random(
                n, target_model, global_context
            )
        if recall + precision == 0:
            f1_score = 0
        else:
            f1_score = 2 * recall * precision / (recall + precision)
    return (
        pos_per_context,
        neg_per_context,
        score,
        recall,
        precision,
        accuracy,
        f1_score,
        regret,
        infeasiblity,
        time_taken,
        t,
        iteration,
        num_nbr,
    )


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

def evaluate_statistics_random(
    n,
    target_model: MaxSatModel,
    context: Context,
    sample_size=1000,
    seed=111,
):
    f1_rand, reg_rand, inf_rand = random_classifier(
        n, target_model, context, sample_size, seed
    )
    return 1,1,1, reg_rand, inf_rand

def evaluate_statistics_sampling(
    n,
    target_model: MaxSatModel,
    learned_model: MaxSatModel,
    context: Context,
    sample_size,
    seed,
):
    recall, precision, accuracy = get_recall_precision_sampling(
        n, target_model, learned_model, context, sample_size, seed
    )
    reg, infeasibility = regret(n, target_model, learned_model, context)
    f1_rand, reg_rand, inf_rand = random_classifier(
        n, target_model, context, sample_size, seed
    )
    return recall, precision, accuracy, reg, infeasibility, f1_rand, reg_rand, inf_rand


def random_classifier(n, target_model, context, sample_size, seed):
    rng = np.random.RandomState(seed)
    tp = 0
    learned_sols = []
    while len(learned_sols) < sample_size:
        instance = rng.rand(n) > 0.5
        # for i in rng.choice(list(context), 1):
        #     instance[abs(i) - 1] = i > 0
        if list(instance) in learned_sols:
            continue
        learned_sols.append(list(instance))
        if label_instance(target_model, instance, context):
            tp += 1
    recall = tp * 100 / sample_size

    sol, cost = solve_weighted_max_sat(n, target_model, context, 1)
    opt_val = get_value(target_model, sol, context)
    avg_regret = 0
    infeasible = 0
    for learned_sol in learned_sols:
        learned_opt_val = get_value(target_model, np.array(learned_sol), context)
        if not learned_opt_val:
            infeasible += 1
        else:
            regret = (opt_val - learned_opt_val) * 100 / opt_val
            avg_regret += regret
    if infeasible < len(learned_sols):
        avg_regret = avg_regret / (len(learned_sols) - infeasible)
    else:
        avg_regret = -1

    f1 = (2 * recall * 50) / (recall + 50)
    return f1, avg_regret, infeasible * 100 / len(learned_sols)


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


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False



def main(args):
    iterations = list(
        it.product(
            args.file,
            args.num_hard,
            args.num_soft,
            args.model_seeds,
            args.num_context,
            args.num_pos,
            args.num_neg,
            args.neg_type,
            args.context_seeds,
            args.method,
            args.cutoff,
            args.noise,
        )
    )
    pool = Pool(args.pool)
    if args.function == "g":
        pool.starmap(generate, [itr[:9] for itr in iterations])

    elif args.function == "e":
        folder_name = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
        os.makedirs(f"results/{folder_name}")
        with open(f"results/{folder_name}/arguments.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)
        csvfile = open(f"results/{folder_name}/evaluation.csv", "w")
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(
            [
                "file",
                "num_hard",
                "num_soft",
                "model_seed",
                "num_context",
                "num_pos",
                "num_neg",
                "neg_type",
                "context_seed",
                "method",
                "max_cutoff",
                "noise",
                "pos_per_context",
                "neg_per_context",
                "score",
                "recall",
                "precision",
                "accuracy",
                "f1_score",
                "regret",
                "infeasiblity",
                "time_taken",
                "cutoff",
                "iterations",
                "neighbours",
            ]
        )
        # print(iterations)
        stats = pool.starmap(evaluate, iterations)
        for i, s in enumerate(stats):
            tmp = list(iterations[i])
            tmp.extend(list(s))
            filewriter.writerow(tmp)
        csvfile.close()
    else:
        pool.starmap(learn, iterations)



logger = logging.getLogger(__name__)
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="g")
    CLI.add_argument("--file", nargs="*", type=str, default=["uf20-01.cnf"])
    CLI.add_argument("--num_hard", nargs="*", type=int, default=[10])
    CLI.add_argument("--num_soft", nargs="*", type=int, default=[10])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111]
    )
    CLI.add_argument("--num_context", nargs="*", type=int, default=[250])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111]
    )
    CLI.add_argument("--num_pos", nargs="*", type=int, default=[2])
    CLI.add_argument("--num_neg", nargs="*", type=int, default=[2])
    CLI.add_argument("--neg_type", nargs="*", type=str, default=["both"])
    CLI.add_argument(
        "--method",
        nargs="*",
        type=str,
        default=["walk_sat"],
    )
    CLI.add_argument("--cutoff", nargs="*", type=int, default=[3600])
    CLI.add_argument("--noise", nargs="*", type=float, default=[0])
    CLI.add_argument("--pool", type=int, default=1)

    args = CLI.parse_args()

    main(args)
    #
    # if args.function == "generate":
    #     generate(args)
    #
    # elif args.function == "learn":
    #     learn(args)
    #
    # elif args.function == "evaluate":
    #     evaluate(args, bl=0)
