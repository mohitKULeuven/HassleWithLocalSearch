from .synthetic import learn_model_MILP
from .benchmark import cnf_param, evaluate, generate
from code.local_search import learn_weighted_max_sat
import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
import itertools as it


def learn(args):
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
    bar = tqdm("Progress", total=iterations, position=0)
    for c, context_seed, method, t, p in it.product(
        args.num_context, args.context_seeds, args.method, args.cutoff, args.noise
    ):
        for cnf_file in args.filename:
            if cnf_file.endswith(".wcnf") or cnf_file.endswith(".cnf"):
                for s in args.per_soft:
                    for seed in args.model_seeds:
                        n, m = cnf_param(args.path + cnf_file, args.num_constraints)
                        param = f"_{cnf_file}_num_constraints_{args.num_constraints}_per_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                        if method == "MILP":
                            try:
                                learn_model_MILP(m, method, t, param, p)
                                bar.update(1)
                            except FileNotFoundError:
                                print("FileNotFound: " + param)
                                continue
                        else:
                            try:
                                learn_model(m, method, t, param, p)
                                bar.update(1)
                            except FileNotFoundError:
                                print("FileNotFound: " + param)
                                continue


def learn_model(num_constraints, method, cutoff, param, p):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    param += f"_method_{method}_cutoff_{cutoff}_noise_{p}_bl"

    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        tqdm.write("Exists: " + param + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]
    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    inf = [True if l == -1 else False for l in labels]
    labels = np.array([True if l == 1 else False for l in labels])

    if p != 0:
        rng = np.random.RandomState(111)
        for i, label in enumerate(labels):
            if rng.random_sample() < p:
                labels[i] = not label

    learn_weighted_max_sat(
        num_constraints,
        3,
        data,
        labels,
        contexts,
        method,
        param,
        inf,
        cutoff_time=cutoff,
    )


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="generate")
    CLI.add_argument("--path", type=str, default="cnfs/3cnf_benchmark/")
    CLI.add_argument("--filename", nargs="*", type=str, default=[])
    CLI.add_argument("--num_constraints", type=int, default=20)
    CLI.add_argument("--per_soft", nargs="*", type=int, default=[50])
    CLI.add_argument("--model_seeds", nargs="*", type=int, default=[111])
    CLI.add_argument("--num_context", nargs="*", type=int, default=[1000])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_pos", type=int, default=2)
    CLI.add_argument("--num_neg", type=int, default=2)
    CLI.add_argument("--sample_size", type=int, default=1000)
    CLI.add_argument("--method", nargs="*", type=str, default=["novelty"])
    CLI.add_argument("--cutoff", nargs="*", type=int, default=[86400])
    CLI.add_argument("--noise", nargs="*", type=float, default=[0])
    CLI.add_argument("--weighted", type=int, default=1)

    args = CLI.parse_args()

    if args.function == "generate":
        generate(args)

    elif args.function == "learn":
        learn(args)

    elif args.function == "evaluate":
        evaluate(args, 1)
