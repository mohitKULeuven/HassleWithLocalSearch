import logging
import pickle
import csv
import argparse
import itertools as it
import numpy as np
from datetime import datetime
import os
import json
import time

from code.type_def import MaxSatModel, Context
from code.generator import generate_models, generate_contexts_and_data
from code.pysat_solver import solve_weighted_max_sat, get_value, label_instance
from code.local_search import learn_weighted_max_sat

from code.milp_learner import learn_weighted_max_sat_MILP
from code.verify import get_recall_precision_wmc


def learn_model_noisy(num_constraints, method, cutoff, param, w, p):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    param += f"_method_{method}_cutoff_{cutoff}_noise_{p}"

    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        print("Exists: " + param + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]
    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    if p != 0:
        rng = np.random.RandomState(111)
        for i, label in enumerate(labels):
            if rng.random_sample() < p:
                labels[i] = not label

    models, scores, time_taken, iterations = learn_weighted_max_sat(
        num_constraints,
        data,
        labels,
        contexts,
        method,
        int(len(labels) * 1),
        w,
        cutoff_time=cutoff,
    )

    for i, score in enumerate(scores):
        scores[i] = scores[i] * 100 / data.shape[0]

    pickle_var["learned_model"] = models
    pickle_var["time_taken"] = time_taken
    pickle_var["score"] = scores
    pickle_var["iterations"] = iterations

    pickle.dump(pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb"))
    print(param + ": " + str(pickle_var["score"][-1]) + "\n")
    return models[-1], time_taken[-1]


def learn(args):
    for n, h, s, seed, c, context_seed, m, t in it.product(
        args.num_vars,
        args.num_hard,
        args.num_soft,
        args.model_seeds,
        args.num_context,
        args.context_seeds,
        args.method,
        args.cutoff,
    ):
        try:
            param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
            for p in args.noise:
                learn_model_noisy(h + s, m, t, param, args.weighted, p)
        except FileNotFoundError:
            continue
