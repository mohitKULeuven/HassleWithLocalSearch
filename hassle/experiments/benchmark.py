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
)
from hassle.pysat_solver import solve_weighted_max_sat, get_value, label_instance
from hassle.verify import get_recall_precision_sampling
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


def generate(path, h, s, seed, nc, num_pos, num_neg, neg_type, c_seed):
    for cnf_file in os.listdir(path):
        if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
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


def learn(path, h, s, seed, c, num_pos, num_neg, neg_type, context_seed, method, t, p):
    for cnf_file in os.listdir(path):
        if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
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
                    continue
            else:
                try:
                    learn_model_sls(m, method, t, param, p, 1)
                except FileNotFoundError:
                    print("FileNotFound: " + param)
                    continue


def evaluate(args, bl):
    folder_name = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
    os.makedirs(f"results/{folder_name}")
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
            "noise_probability",
            "iterations",
            "bl",
        ]
    )
    if not args.filename:
        args.filename = os.listdir(args.path)
    iterations = (
        len(args.filename)
        * len(args.per_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
        * len(args.method)
        * len(args.cutoff)
        * len(args.noise)
    )
    bar = tqdm(total=iterations)
    for s, seed in it.product(args.per_soft, args.model_seeds):
        for cnf_file in args.filename:
            if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                n, m = cnf_param(args.path + cnf_file, args.num_constraints)
                num_soft = int(m * s / 100)
                param = f"_{cnf_file}_num_constraints_{args.num_constraints}_per_soft_{s}_model_seed_{seed}"
                pickle_var = pickle.load(
                    open("pickles/target_model/" + param + ".pickle", "rb")
                )
                target_model = pickle_var["true_model"]
                max_t = max(args.cutoff)
                for c, context_seed in it.product(args.num_context, args.context_seeds):
                    tag_cnd = (
                        param
                        + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                    )
                    if args.neg_type:
                        tag_cnd = (
                            param
                            + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_neg_type_{args.neg_type}_context_seed_{context_seed}"
                        )
                    pickle_cnd = pickle.load(
                        open("pickles/contexts_and_data/" + tag_cnd + ".pickle", "rb")
                    )
                    for method, t, p in it.product(
                        args.method, args.cutoff, args.noise
                    ):
                        tag_lm = tag_cnd + f"_method_{method}_cutoff_{max_t}_noise_{p}"
                        if bl == 1:
                            tag_lm += "_bl"
                        if args.weighted == 0:
                            pickle_var = pickle.load(
                                open(
                                    "pickles/learned_model/" + tag_lm + ".pickle", "rb"
                                )
                            )
                        else:
                            pickle_var = pickle.load(
                                open(
                                    "pickles/learned_model/" + tag_lm + ".pickle", "rb"
                                )
                            )
                        i = 0
                        if t < max_t:
                            for i, cutoff in enumerate(pickle_var["time_taken"]):
                                if cutoff > t:
                                    break
                        learned_model = pickle_var["learned_model"][i - 1]
                        time_taken = pickle_var["time_taken"][i - 1]
                        score = pickle_var["score"][i - 1]
                        iteration = pickle_var["iterations"][i - 1]

                        contexts = pickle_cnd["contexts"]
                        global_context = set()
                        for context in contexts:
                            global_context.update(context)

                        recall, precision, accuracy, regret = -1, -1, -1, -1
                        if learned_model:
                            recall, precision, accuracy, regret, infeasiblity = evaluate_statistics(
                                n,
                                target_model,
                                learned_model,
                                global_context
                                # args.sample_size,
                                # seed,
                            )
                            f1_random, reg_random, inf_random = random_classifier(
                                n, target_model, global_context, args.sample_size, seed
                            )
                        f1_score = 0
                        if recall + precision != 0:
                            f1_score = 2 * recall * precision / (recall + precision)
                        labels = [
                            True if l == 1 else False for l in pickle_cnd["labels"]
                        ]
                        if c == 0:
                            c = 1
                        pos_per_context = labels.count(True) / c
                        neg_per_context = labels.count(False) / c

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
                                f1_random,
                                reg_random,
                                inf_random,
                                time_taken,
                                t,
                                p,
                                iteration,
                                bl,
                            ]
                        )
                        bar.update(1)
    csvfile.close()


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
        for i in rng.choice(list(context), 1):
            instance[abs(i) - 1] = i > 0
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
            args.path,
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
                "num_vars",
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
                "use_context",
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
    CLI.add_argument("--path", type=str, default=["cnfs/3cnf_benchmark/"])
    CLI.add_argument("--num_hard", nargs="*", type=int, default=[10])
    CLI.add_argument("--num_soft", nargs="*", type=int, default=[10])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_context", nargs="*", type=int, default=[250, 500, 1000])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
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
