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
from tqdm import tqdm

from code.type_def import MaxSatModel, Context
from code.generator import generate_models, generate_contexts_and_data
from code.pysat_solver import solve_weighted_max_sat, get_value, label_instance
from code.local_search import learn_weighted_max_sat

from code.milp_learner import learn_weighted_max_sat_MILP
from code.verify import get_recall_precision_wmc


def generate(args):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
    )
    bar = tqdm(total=iterations)
    # bar = progressbar.ProgressBar(max_value=iterations, redirect_stdout=True)
    for n, h, s, seed in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds
    ):
        model, param = generate_models(n, int(n / 2), h, s, seed)

        for c, context_seed in it.product(args.num_context, args.context_seeds):
            tag = generate_contexts_and_data(
                n, model, c, args.num_pos, args.num_neg, param, context_seed
            )
            tqdm.write(tag)
            bar.update(1)


def learn(args):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
        * len(args.method)
        * len(args.cutoff)
        * len(args.noise)
    )
    bar = tqdm(total=iterations)
    for n, h, s, seed, c, context_seed, m, t, p in it.product(
        args.num_vars,
        args.num_hard,
        args.num_soft,
        args.model_seeds,
        args.num_context,
        args.context_seeds,
        args.method,
        args.cutoff,
        args.noise,
    ):
        if m == "MILP":
            try:
                param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                learn_model_MILP(h + s, m, t, param, p)
            except FileNotFoundError:
                continue
        else:
            try:
                param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                learn_model(h + s, m, t, param, args.weighted, p)
            except FileNotFoundError:
                continue
        bar.update(1)


def evaluate(args):
    iterations = (
        len(args.num_vars)
        * len(args.num_hard)
        * len(args.num_soft)
        * len(args.model_seeds)
        * len(args.num_context)
        * len(args.context_seeds)
        * len(args.method)
        * len(args.cutoff)
        * len(args.noise)
    )
    bar = tqdm(total=iterations)
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
            "time_taken",
            "cutoff",
            "noise_probability",
            "iterations",
        ]
    )
    for n, h, s, seed in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds
    ):
        param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}"
        pickle_var = pickle.load(
            open("pickles/target_model/" + param + ".pickle", "rb")
        )
        target_model = pickle_var["true_model"]
        max_t = max(args.cutoff)
        for c, context_seed, m, p in it.product(
            args.num_context, args.context_seeds, args.method, args.noise
        ):
            tag = (
                param
                + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}_method_{m}_cutoff_{max_t}_noise_{p}"
            )
            if m == "MILP":
                pickle_var = pickle.load(
                    open("pickles/learned_model/" + tag + ".pickle", "rb")
                )
            else:
                pickle_var = pickle.load(
                    open("pickles/learned_model/" + tag + ".pickle", "rb")
                )
            if c == 0:
                pos_per_context = pickle_var["labels"].count(True)
                neg_per_context = pickle_var["labels"].count(False)
            else:
                pos_per_context = pickle_var["labels"].count(True) / c
                neg_per_context = pickle_var["labels"].count(False) / c
            last_index = -2
            recall, precision, accuracy = (-1, -1, -1)
            regret, infeasiblity, f1_score = (-1, -1, -1)
            for t in args.cutoff:
                index = get_learned_model(pickle_var["time_taken"], max_t, t)
                learned_model = None
                time_taken = t
                iteration = 0
                score = -1
                if index is not None:
                    learned_model = pickle_var["learned_model"][index]
                    time_taken = pickle_var["time_taken"][index]
                    iteration = pickle_var["iterations"][index]
                    if learned_model:
                        score = pickle_var["score"][index]

                if index == last_index:
                    # print(score, accuracy, f1_score, infeasiblity, regret)
                    filewriter.writerow(
                        [
                            n,
                            h,
                            s,
                            seed,
                            c,
                            context_seed,
                            args.num_pos,
                            args.num_neg,
                            pos_per_context,
                            neg_per_context,
                            m,
                            score,
                            recall,
                            precision,
                            accuracy,
                            f1_score,
                            regret,
                            infeasiblity,
                            time_taken,
                            t,
                            p,
                            iteration,
                        ]
                    )
                    bar.update(1)
                    continue
                last_index = index

                contexts = pickle_var["contexts"]
                global_context = set()
                for context in contexts:
                    global_context.update(context)
                if learned_model:
                    recall, precision, accuracy, regret, infeasiblity = evaluate_statistics(
                        n, target_model, learned_model, global_context
                    )
                f1_score = 2 * recall * precision / (recall + precision)
                # print(score, accuracy, f1_score, infeasiblity, regret)
                filewriter.writerow(
                    [
                        n,
                        h,
                        s,
                        seed,
                        c,
                        context_seed,
                        args.num_pos,
                        args.num_neg,
                        pos_per_context,
                        neg_per_context,
                        m,
                        score,
                        recall,
                        precision,
                        accuracy,
                        f1_score,
                        regret,
                        infeasiblity,
                        time_taken,
                        t,
                        p,
                        iteration,
                    ]
                )
                bar.update(1)

    csvfile.close()


def learn_model(num_constraints, method, cutoff, param, w, p):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    param += f"_method_{method}_cutoff_{cutoff}_noise_{p}"

    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        tqdm.write("Exists: " + param + "\n")
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

    if "cnf" in param:
        pickle_var["learned_model"] = [models[-1]]
        pickle_var["time_taken"] = [time_taken[-1]]
        pickle_var["score"] = [scores[-1]]
        pickle_var["iterations"] = iterations
    else:
        pickle_var["learned_model"] = models
        pickle_var["time_taken"] = time_taken
        pickle_var["score"] = scores
        pickle_var["iterations"] = iterations
    if not os.path.exists("pickles/learned_model"):
        os.makedirs("pickles/learned_model")
    pickle.dump(pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb"))
    # tqdm.write(param + ": " + str(pickle_var["score"][-1]) + "\n")
    return models[-1], time_taken[-1]


def learn_model_MILP(num_constraints, method, cutoff, param, p):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )

    param += f"_method_{method}_cutoff_{cutoff}_noise_{p}"
    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        tqdm.write("Exists: " + param + ": " + str(pickle_var["score"]) + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]

    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    if p != 0:
        rng = np.random.RandomState(111)
        for i, label in enumerate(labels):
            if rng.random_sample() < p:
                labels[i] = not label

    start = time.time()
    learned_model = learn_weighted_max_sat_MILP(
        num_constraints, data, labels, contexts, cutoff
    )
    end = time.time()

    score = 0
    if learned_model:
        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(learned_model, instance, contexts[k])
            if label == learned_label:
                score += 1

    pickle_var["learned_model"] = [learned_model]
    pickle_var["time_taken"] = [end - start]
    pickle_var["score"] = [score * 100 / data.shape[0]]
    if not os.path.exists("pickles/learned_model"):
        os.makedirs("pickles/learned_model")
    pickle.dump(pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb"))
    # tqdm.write(param + ": " + str(pickle_var["score"]) + "\n")
    return learned_model, end - start


def evaluate_statistics(
    n, target_model: MaxSatModel, learned_model: MaxSatModel, context: Context
):
    recall, precision, accuracy = get_recall_precision_wmc(
        n, target_model, learned_model, context
    )
    reg, infeasiblity = regret(n, target_model, learned_model, context)

    return recall, precision, accuracy, reg, infeasiblity


def regret(n, target_model, learned_model, context):

    sol, cost = solve_weighted_max_sat(n, target_model, context, 1)
    opt_val = get_value(target_model, sol)
    avg_regret = 0
    infeasible = 0
    learned_sols, cost = solve_weighted_max_sat(n, learned_model, context, 100)
    for learned_sol in learned_sols:
        learned_opt_val = get_value(target_model, learned_sol)
        if not learned_opt_val:
            infeasible += 1
        else:
            regret = (opt_val - learned_opt_val) * 100 / opt_val
            avg_regret += regret
    if infeasible < len(learned_sols):
        avg_regret = avg_regret / (len(learned_sols) - infeasible)
    else:
        avg_regret = -1

    return avg_regret, infeasible * 100 / len(learned_sols)


def get_learned_model(time_taken, max_cutoff, cutoff):
    if cutoff == max_cutoff:
        return -1
    elif cutoff < max_cutoff:
        ind = 0
        for index, t in enumerate(time_taken):
            if t <= cutoff:
                ind = 1
            elif cutoff < t and ind == 1:
                break
            elif cutoff < t and ind == 0:
                return None
    return index - 1


logger = logging.getLogger(__name__)
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="evaluate")
    CLI.add_argument("--num_vars", nargs="*", type=int, default=[5, 10, 20])
    CLI.add_argument("--num_hard", nargs="*", type=int, default=[5, 10, 20])
    CLI.add_argument("--num_soft", nargs="*", type=int, default=[5])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument(
        "--num_context", nargs="*", type=int, default=[10, 25, 50, 100, 150]
    )
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_pos", type=int, default=2)
    CLI.add_argument("--num_neg", type=int, default=2)
    CLI.add_argument(
        "--method",
        nargs="*",
        type=str,
        default=[
            "walk_sat",
            "novelty",
            "novelty_plus",
            "adaptive_novelty_plus",
            "MILP",
        ],
    )
    CLI.add_argument(
        "--cutoff", nargs="*", type=int, default=[60, 300, 600, 900, 1200, 1500, 1800]
    )
    CLI.add_argument("--noise", nargs="*", type=float, default=[0.05, 0.1, 0.2])
    CLI.add_argument("--weighted", type=int, default=1)

    args = CLI.parse_args()

    if args.function == "generate":
        generate(args)

    elif args.function == "learn":
        learn(args)

    elif args.function == "evaluate":
        evaluate(args)