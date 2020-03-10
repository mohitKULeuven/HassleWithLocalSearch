import logging
import pickle
import csv
import argparse
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import time

from generator import generate_models, generate_contexts_and_data
from pysat_solver import solve_weighted_max_sat, get_value
from local_search import learn_weighted_max_sat
from milp_learner import learn_weighted_max_sat_MILP,label_instance
# from dask import delayed


def learn_model(
    n,
    max_clause_length,
    num_constraints,
    method,
    cutoff, param,w
):
#    print(param)
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    param += f"_method_{method}_cutoff_{cutoff}"
    if w==0 and os.path.exists("pickles/bin_weight/learned_model" + param + ".pickle"):
        pickle_var=pickle.load(
                open("pickles/bin_weight/learned_model" + param + ".pickle", "rb")
        )
        return pickle_var["learned_model"], pickle_var["time_taken"]
    elif w==1 and os.path.exists("pickles/con_weight/learned_model" + param + ".pickle"):
        pickle_var=pickle.load(
                open("pickles/con_weight/learned_model" + param + ".pickle", "rb")
        )
        return pickle_var["learned_model"], pickle_var["time_taken"]
    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    model, score, time_taken, scores, best_scores = learn_weighted_max_sat(
        num_constraints,
        data,
        labels,
        contexts,
        method,
        len(labels),w,
        cutoff_time=cutoff
    )

    pickle_var["learned_model"] = model
    pickle_var["time_taken"] = time_taken
    pickle_var["score"] = score
    pickle_var["scores"] = scores
    pickle_var["best_scores"] = best_scores
    if w==0:
        pickle.dump(pickle_var, open("pickles/bin_weight/learned_model" + param + ".pickle", "wb"))
    else:
        pickle.dump(pickle_var, open("pickles/con_weight/learned_model" + param + ".pickle", "wb"))
    print(param +"\n")
    return model, time_taken


def learn_model_MILP(
    n,
    max_clause_length,
    num_constraints,
    method,
    cutoff, param,w
):
#    print(param)
    pickle_var = pickle.load(
        open("pickles/contexts_and_data/" + param + ".pickle", "rb")
    )
    param += f"_method_{method}_cutoff_{cutoff}"
    if os.path.exists("pickles/bin_weight/learned_model" + param + ".pickle"):
        pickle_var=pickle.load(
                open("pickles/bin_weight/learned_model" + param + ".pickle", "rb")
        )
        return pickle_var["learned_model"], pickle_var["time_taken"]
   
    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]

    start = time.time()
    learned_model = learn_weighted_max_sat_MILP(
        num_constraints,data,labels,contexts,cutoff
    )
    end = time.time()
    
    score = 0
    if learned_model:
        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(
                learned_model, instance, contexts[k]
            )
            if label != learned_label:
                score += 1


    pickle_var["learned_model_MILP"] = learned_model
    pickle_var["time_taken"] = end-start
    pickle_var["score"] = score
    
    pickle.dump(pickle_var, open("pickles/bin_weight/learned_model" + param + ".pickle", "wb"))
    print(param +"\n")
    return learned_model, end-start



def evaluate_statistics(n, target_model, learned_model, sample_size):
    sample_target_model,cst = solve_weighted_max_sat(n, target_model, [], sample_size)
#    sample_target_model,labels = generate_data(n, target_model, [], sample_size, seed)
#    print(sample_target_model)
#    return
    recall = eval_recall(n, sample_target_model, learned_model)
    sample_learned_model,cst = solve_weighted_max_sat(n, learned_model, [], sample_size)
    precision, regret = eval_precision_regret(
        n, sample_learned_model, target_model, learned_model
    )
    return recall, precision, regret


def eval_recall(n, sample, model):
    recall = 0
    sol, cost = solve_weighted_max_sat(n, model, [], 1)
    if not sol:
        return recall
    opt_val = get_value(model, sol)
    for example in sample:
        val = get_value(model, example)
        if val is not None and val == opt_val:
            recall += 1
    return recall * 100 / len(sample)


def eval_precision_regret(n, sample, target_model, learned_model):
    precision = 0
    learned_sol, cost = solve_weighted_max_sat(n, learned_model, [], 1)
    learned_opt_val = get_value(target_model, learned_sol)

    sol, cost = solve_weighted_max_sat(n, target_model, [], 1)
    opt_val = get_value(target_model, sol)

    regret = (opt_val - learned_opt_val)* 100 / opt_val if learned_opt_val else -1

    for example in sample:
        val = get_value(target_model, example)
        if val is not None and val == opt_val:
            precision += 1
    return precision * 100 / len(sample), regret



def generate(args):
    for n, h, s, seed in it.product(
        args.num_vars, args.num_hard, args.num_soft, args.model_seeds
    ):
        model, param = generate_models(n, int(n/2), h, s, seed)

        for c, context_seed, d in it.product(
            args.num_context, args.context_seeds, args.data_size
        ):
            tag = generate_contexts_and_data(n, model, c, d, param, context_seed)
            print(tag)

def learn(args):
    for n, h, s, seed, c, context_seed, d, m, t in it.product(
        args.num_vars,args.num_hard,args.num_soft,args.model_seeds,
        args.num_context,args.context_seeds,args.data_size,args.method,
        args.cutoff,
    ):
        if m=="MILP":
            try:
                param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}"
                learn_model_MILP(n, n, h + s, m, t, param)
            except FileNotFoundError:
                continue
        else:
            try:
                param = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}"
                learn_model(n, n, h + s, m, t, param,args.weighted)
            except FileNotFoundError:
                continue

def evaluate(args):
    folder_name=datetime.now().strftime("%d-%m-%y (%H:%M:%S.%f)")
    os.mkdir(f"results/{folder_name}")
    with open(f"results/{folder_name}/arguments.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    csvfile = open(f"results/{folder_name}/evaluation.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "num_vars","num_hard","num_soft","model_seed","num_context",
            "context_seed","data_size","pos_per_context","neg_per_context",
            "method","score","recall","precision","regret","time_taken","cutoff"
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
        for c, context_seed, d, m, t in it.product(
            args.num_context,args.context_seeds,
            args.data_size,args.method,args.cutoff,
        ):
            tag = (
                param
                + f"_num_context_{c}_num_data_{d}_context_seed_{context_seed}_method_{m}_cutoff_{t}"
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
#            print(target_model)
#            print(learned_model)
            time_taken = pickle_var["time_taken"]
            score = pickle_var["score"]
            recall, precision, regret = evaluate_statistics(
                n, target_model, learned_model, args.sample_size
            )
#            return
            pos_per_context=pickle_var["labels"].count(True)/c
            neg_per_context=pickle_var["labels"].count(False)/c
            print(n, h, s, d, score, recall, precision, regret)
            filewriter.writerow(
                [
                    n,h,s,seed,c,context_seed,d,pos_per_context,neg_per_context,
                    m,score,recall,precision,regret,time_taken,t
                ]
            )
    csvfile.close()

def avg_training_score(args):
    for method in args.method:
        for t in args.cutoff:
            score = []
            for n, h, s, seed, c, context_seed, d in it.product(
                args.num_vars, args.num_hard, args.num_soft,args.model_seeds,
                args.num_context,args.context_seeds,args.data_size
            ):
                try:
                    tag = f"_n_{n}_max_clause_length_{n}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}_method_{method}_cutoff_{t}"
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
    plt.ylim(60,100)
    for method in args.method:
        avg_score = []
        for n, h, s, seed, c, context_seed, d in it.product(
            args.num_vars, args.num_hard, args.num_soft,args.model_seeds,
            args.num_context,args.context_seeds,args.data_size
        ):
            score = []
#            cutoff_time = []
            for t in args.cutoff:
                
                try:
                    tag = f"_n_{n}_max_clause_length_{int(n/2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}_method_{method}_cutoff_{t}"
                    if args.weighted==0:
                        pickle_var = pickle.load(
                            open("pickles/bin_weight/learned_model" + tag + ".pickle", "rb")
                        )
                    else:
                        pickle_var = pickle.load(
                            open("pickles/con_weight/learned_model" + tag + ".pickle", "rb")
                        )
                    score.append(pickle_var["score"])
#                    cutoff_time.append(t)
                except FileNotFoundError:
                    continue
            if score:
                avg_score.append(score)
#                print(method, score)
#                plt.plot(cutoff_time, score, label=method)
#                plt.legend(loc="lower right")
#                plt.draw()
#                sav=1
        avg_score=np.array(avg_score)
        y=np.average(avg_score,axis=0)
#        y_err=[elem/np.sqrt(len(avg_score[i])) for i,elem in enumerate(np.std(avg_score,axis=0))]
        print(method,y)
#        plt.errorbar(args.cutoff, np.average(avg_score,axis=0), 
#                     yerr=np.std(avg_score,axis=0), label=method)
        plt.plot(args.cutoff, np.average(avg_score,axis=0), label=method)
#        plt.fill_between(args.cutoff, y-y_err, y+y_err,alpha=0.3)
        plt.legend(loc="lower right")
        plt.draw()
    tag = f"results/synthetic_evaluation_over_score_neg.png"
    fig.savefig(tag)



logger = logging.getLogger(__name__)
if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    args = parser.parse_args()
#    with open('results/26-02-20 (13:36:06.614870)/arguments.txt', 'r') as f:
#        args.__dict__ = json.load(f)
#    
#    print(args)
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--function", type=str, default="evaluate")
    CLI.add_argument("--num_vars", nargs="*", type=int, default=[5,10,20])
    CLI.add_argument("--num_hard", nargs="*", type=int, default=[5, 10, 20])
    CLI.add_argument("--num_soft", nargs="*", type=int, default=[5])
    CLI.add_argument(
        "--model_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--num_context", nargs="*", type=int, default=[5, 10])
    CLI.add_argument(
        "--context_seeds", nargs="*", type=int, default=[111, 222, 333, 444, 555]
    )
    CLI.add_argument("--data_size", nargs="*", type=int, default=[2, 5])
    CLI.add_argument("--sample_size", type=int, default=1000)
    CLI.add_argument(
        "--method",
        nargs="*",
        type=str,
        default=["walk_sat", "novelty", "novelty_plus", "adaptive_novelty_plus"],
    )
    CLI.add_argument("--cutoff", nargs="*", type=int, default=[2, 10, 60])
    CLI.add_argument("--weighted", type=int, default=1)
    
    args = CLI.parse_args()
    
#    print(args)

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