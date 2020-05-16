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

from type_def import MaxSatModel, Context
from generator import generate_models, generate_contexts_and_data
from pysat_solver import solve_weighted_max_sat, get_value, label_instance
from local_search import learn_weighted_max_sat
#from milp_learner import learn_weighted_max_sat_MILP
from verify import get_recall_precision


def learn_model(n, max_clause_length, num_constraints, method, cutoff, param, w):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    param += f"_method_{method}_cutoff_{cutoff}"

    if os.path.exists(
        "pickles/learned_model/" + param + ".pickle"
    ):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        print("Exists: "+param + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]
    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    models, scores, time_taken,iterations = learn_weighted_max_sat(
        num_constraints,
        data,
        labels,
        contexts,
        method,
        int(len(labels) * 1),
        w,
        cutoff_time=cutoff,
    )
    
    for i,score in enumerate(scores):
        scores[i]=scores[i] * 100 / data.shape[0]

    pickle_var["learned_model"] = models
    pickle_var["time_taken"] = time_taken
    pickle_var["score"] = scores
    pickle_var["iterations"] = iterations

    pickle.dump(
        pickle_var,
        open("pickles/learned_model/" + param + ".pickle", "wb"),
    )
    print(param + ": " + str(pickle_var["score"][-1]) + "\n")
    return models[-1], time_taken[-1]


def learn_model_MILP(n, max_clause_length, num_constraints, method, 
                     cutoff, max_cutoff, param):
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    
    param += f"_method_{method}_cutoff_{cutoff}"
    if os.path.exists("pickles/learned_model/" + param + ".pickle"):
        pickle_var = pickle.load(
            open("pickles/learned_model/" + param + ".pickle", "rb")
        )
        print("Exists: "+param + ": " + str(pickle_var["score"]) + "\n")
        return pickle_var["learned_model"], pickle_var["time_taken"]

    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

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

    pickle.dump(
        pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb")
    )
    print(param + ": " + str(pickle_var["score"]) + "\n")
    return learned_model, end - start


def evaluate_statistics_sampling(
    n,
    target_model: MaxSatModel,
    learned_model: MaxSatModel,
    context: Context,
    sample_size,
    seed,
):
    tp1, fn = recall_precision(
        n, target_model, learned_model, context, sample_size, seed
    )
    tp2, fp = recall_precision(
        n, learned_model, target_model, context, sample_size, seed
    )
    recall = tp1 * 100 / (tp1 + fn)
    precision = tp2 * 100 / (tp2 + fp)
    reg,infeasiblity = regret(n, target_model, learned_model,context)

    rng = np.random.RandomState(seed)
    learned_neg_count = 0
    tn = 0
    neg_sample = []
    max_tries = 10 * sample_size
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        for i in rng.choice(list(context), 1):
            instance[abs(i) - 1] = i > 0
        if list(instance) in neg_sample:
            continue
        neg_sample.append(list(instance))
        if not label_instance(learned_model, instance, context):
            learned_neg_count += 1
            if not label_instance(target_model, instance, context):
                tn += 1
            if learned_neg_count == sample_size:
                break
    accuracy = (tp1 + tp2 + tn) * 100 / (tp1 + tp2 + fn + fp + learned_neg_count)
    f1_random, reg_random, inf_random=random_classifier(n, target_model, context, sample_size, seed)
    return recall, precision, accuracy, reg, infeasiblity, f1_random, reg_random, inf_random


def evaluate_statistics(
    n, target_model: MaxSatModel, learned_model: MaxSatModel, context: Context
):
    recall, precision, accuracy = get_recall_precision(
        n, target_model, learned_model, context
    )
    reg,infeasiblity=regret(n, target_model, learned_model,context)

    return recall, precision, accuracy, reg,infeasiblity


def recall_precision(n, model1, model2, context, sample_size, seed):
    rng = np.random.RandomState(seed)
    tp = 0
    sample_model1 = []
    tmp_data, cst = solve_weighted_max_sat(n, model1, context, sample_size * 10)
    if len(tmp_data) > sample_size:
        indices = list(rng.choice(range(len(tmp_data)), sample_size, replace=False))
        for i in indices:
            sample_model1.append(tmp_data[i])
    else:
        sample_model1 = tmp_data
    sol, cost = solve_weighted_max_sat(n, model2, context, 1)
    if not sol:
        return tp, len(sample_model1) - tp
    opt_val = get_value(model2, sol)
    for example in sample_model1:
        val = get_value(model2, example)
        if val is not None and val == opt_val:
            tp += 1
    return tp, len(sample_model1) - tp

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
            tp+=1
    recall=tp*100/sample_size
    
    sol, cost = solve_weighted_max_sat(n, target_model, context, 1)
    opt_val = get_value(target_model, sol)
    avg_regret=0
    infeasible=0
    for learned_sol in learned_sols:
        learned_opt_val = get_value(target_model, learned_sol)
        if not learned_opt_val:
            infeasible+=1
        else:
            regret = (opt_val - learned_opt_val) * 100 / opt_val
            avg_regret+=regret
    if infeasible<len(learned_sols):
        avg_regret=avg_regret/(len(learned_sols)-infeasible)
    else:
        avg_regret=-1
    
    f1=(2*recall*50)/(recall+50)
    return f1, avg_regret,infeasible*100/len(learned_sols)


def regret(n, target_model, learned_model,context):
    
    sol, cost = solve_weighted_max_sat(n, target_model, context, 1)
    opt_val = get_value(target_model, sol)
    avg_regret=0
    infeasible=0
    learned_sols, cost = solve_weighted_max_sat(n, learned_model, context, 100)
    for learned_sol in learned_sols:
        learned_opt_val = get_value(target_model, learned_sol)
        if not learned_opt_val:
            infeasible+=1
        else:
            regret = (opt_val - learned_opt_val) * 100 / opt_val
            avg_regret+=regret
    if infeasible<len(learned_sols):
        avg_regret=avg_regret/(len(learned_sols)-infeasible)
    else:
        avg_regret=-1
        
    return avg_regret,infeasible*100/len(learned_sols)
    


def generate(args):
    for n, h, s, seed in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds
    ):
        model, param = generate_models(n, int(n / 2), h, s, seed)

        for c, context_seed in it.product(args.num_context, args.context_seeds):
            tag = generate_contexts_and_data(
                n, model, c, args.num_pos, args.num_neg, param, context_seed
            )
            print(tag)


def learn(args):
    max_t=max(args.cutoff)
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
        if m == "MILP":
            try:
                param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                learn_model_MILP(n, n, h + s, m, t,max_t, param)
#                print(param)
            except FileNotFoundError:
                continue
        else:
            try:
                param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}"
                learn_model(n, n, h + s, m, t, param, args.weighted)
#                print(param)
            except FileNotFoundError:
                continue

def get_learned_model(time_taken,max_cutoff,cutoff):
    if cutoff==max_cutoff:
        return -1
    elif cutoff<max_cutoff:
        ind=0
        for index,t in enumerate(time_taken):
#            print(index,t)
            if t<=cutoff:
                ind=1
            elif cutoff<t and ind==1:
                break
            elif cutoff<t and ind==0:
                return None
    return index-1

def evaluate(args):
    folder_name = datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
    os.mkdir(f"results/{folder_name}")
    with open(f"results/{folder_name}/arguments.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    csvfile = open(f"results/{folder_name}/evaluation.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "num_vars","num_hard","num_soft","model_seed","num_context",
            "context_seed","num_pos","num_neg","pos_per_context","neg_per_context",
            "method","score","recall","precision","accuracy","f1_score",
            "regret","infeasiblity","time_taken","cutoff",
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
        max_t=max(args.cutoff)
        for c, context_seed, m in it.product(
            args.num_context, args.context_seeds, args.method
        ):
            tag = (
                param
                + f"_num_context_{c}_num_pos_{args.num_pos}_num_neg_{args.num_neg}_context_seed_{context_seed}_method_{m}_cutoff_{max_t}"
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
            last_index=-2
            recall,precision,accuracy,regret,infeasiblity,f1_score = -1,-1,-1,-1,-1,-1
            for t in args.cutoff:
                index=get_learned_model(pickle_var["time_taken"],max_t,t)
                learned_model = None
                time_taken = t
                score = -1
                if index is not None:
                    learned_model = pickle_var["learned_model"][index]
                    if m!="MILP" and n==10 and h==10:
                        learned_model=learned_model.maxSatModel()
                    time_taken = pickle_var["time_taken"][index]
                    if learned_model:
                        score = pickle_var["score"][index]
                
                if index==last_index:
                    print(c, m, t, score, accuracy, f1_score, infeasiblity, regret)
                    filewriter.writerow(
                        [
                            n,h,s,seed,c,context_seed,args.num_pos,args.num_neg,
                            pos_per_context,neg_per_context,m,score,recall,precision,
                            accuracy,f1_score,regret,infeasiblity,time_taken,t,
                        ]
                    )
                    continue
                last_index=index
                
                contexts = pickle_var["contexts"]
                global_context = set()
                for context in contexts:
                    global_context.update(context)
                if learned_model:
                    recall, precision, accuracy, regret,infeasiblity = evaluate_statistics(
                        n, target_model, learned_model, global_context
                    )
                f1_score = 2 * recall * precision / (recall + precision)
                
                print(c, m, t, score, accuracy, f1_score, infeasiblity, regret)
                filewriter.writerow(
                    [
                        n,h,s,seed,c,context_seed,args.num_pos,args.num_neg,
                        pos_per_context,neg_per_context,m,score,recall,precision,
                        accuracy,f1_score,regret,infeasiblity,time_taken,t,
                    ]
                )
    csvfile.close()


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
    CLI.add_argument("--sample_size", type=int, default=1000)
    CLI.add_argument(
        "--method",
        nargs="*",
        type=str,
        default=[
            "walk_sat",
            "novelty",
            "novelty_plus",
            "adaptive_novelty_plus",
            "MILP"
        ],
    )
    CLI.add_argument(
        "--cutoff", nargs="*", type=int, default=[60, 300, 600, 900, 1200, 1500, 1800]
    )
    CLI.add_argument("--weighted", type=int, default=1)

    args = CLI.parse_args()

    if args.function == "generate":
        generate(args)

    elif args.function == "learn":
        learn(args)

    elif args.function == "evaluate":
        evaluate(args)
