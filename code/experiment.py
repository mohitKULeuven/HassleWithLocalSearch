import logging
import pickle
import csv
import argparse
import itertools as it
import numpy as np

# import os

from generator import generate_models,generate_contexts_and_data, generate_data
from pysat_solver import solve_weighted_max_sat, get_value
from local_search import learn_weighted_max_sat


def learn_model(n, max_clause_length,num_hard,num_soft,model_seed,
                num_context,context_seed,num_data):
    
    param = f"_n_{n}_max_clause_length_{max_clause_length}_num_hard_{num_hard}_num_soft_{num_soft}_model_seed_{model_seed}"
    param += f"_num_context_{num_context}_num_data_{num_data}_context_seed_{context_seed}"
    pickle_var = pickle.load(open("pickles/contexts_and_data" + param + ".pickle", "rb"))
    data = np.array(pickle_var["data"])
    labels = np.array(pickle_var["labels"])
    contexts = pickle_var["contexts"]
    
#    start=time.time()
#    print(data)
#    print(labels)
#    print(contexts)
    learned_model,score,time_taken = learn_weighted_max_sat(num_hard+num_soft, data, labels, contexts,data.shape[0],0.1,5,1)
#    time_taken=time.time()-start
    
    pickle_var["learned_model"] = learned_model
    pickle_var["time_taken"] = time_taken
    pickle_var["score"] = score
    pickle.dump(pickle_var, open("pickles/learned_model" + param + ".pickle", "wb"))
    return learned_model,time_taken


def evaluate_statistics(n,target_model,learned_model,sample_size,seed):
    sample_target_model=generate_data(n, target_model, [], sample_size,seed)
    recall=eval_recall(n,sample_target_model,learned_model)
    sample_learned_model=generate_data(n, learned_model, [], sample_size,seed)
    precision,regret=eval_precision_regret(n,sample_learned_model,target_model,learned_model)
    return recall,precision,regret



def eval_recall(n,sample,model):
    recall=0
    sol,cost=solve_weighted_max_sat(n, model, [], 1)
    if not sol:
        return recall
    opt_val = get_value(model,sol)
    for example in sample:
        val=get_value(model,example)
        if val is not None and val==opt_val:
            recall+=1
    return recall*100/len(sample)

def eval_precision_regret(n,sample,target_model,learned_model):
    precision=0
    learned_sol,cost=solve_weighted_max_sat(n, learned_model, [], 1)
    learned_opt_val = get_value(target_model,learned_sol)
    
    sol,cost=solve_weighted_max_sat(n, target_model, [], 1)
    opt_val = get_value(target_model,sol)
    
    regret = opt_val - learned_opt_val if learned_opt_val else -1
    
    for example in sample:
        val=get_value(target_model,example)
        if val is not None and val==opt_val:
            precision+=1
    return precision*100/len(sample),regret*100/opt_val
            
    
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--function",
        type=str,
        default="evaluate",  
    )
    CLI.add_argument(
        "--num_vars",  
        nargs="*",  
        type=int,
        default=[5],  
    )
    CLI.add_argument(
        "--num_hard",  
        nargs="*",  
        type=int,
        default=[2, 4, 6],  
    )
    CLI.add_argument(
        "--num_soft",  
        nargs="*",  
        type=int,
        default=[2, 4, 6],  
    )
    CLI.add_argument(
        "--model_seeds",  
        nargs="*",  
        type=int,
        default=[111],  
    )
    CLI.add_argument(
        "--num_context",  
        nargs="*",  
        type=int,
        default=[2,4,6],  
    )
    CLI.add_argument(
        "--context_seeds",  
        nargs="*",  
        type=int,
        default=[111],  
    )
    CLI.add_argument(
        "--data_size",  
        nargs="*",  
        type=int,
        default=[2,4,6],  
    )
    CLI.add_argument(
        "--sample_size",  
        type=int,
        default=1000,  
    )
    args = CLI.parse_args()
    
    if args.function=="generate":
        for n,h,s,seed in it.product(args.num_vars,args.num_hard,
                                     args.num_soft,args.model_seeds):
            model,param=generate_models(n,n,h,s,seed)
            for c,context_seed,d in it.product(args.num_context,
                                               args.context_seeds,args.data_size):
                generate_contexts_and_data(n,model,c,d,param,context_seed)
                print(f"_n_{n}_max_clause_length_{n}_num_hard_{h}_num_soft_{s}_model_seed_{seed}_num_context_{c}_num_data_{d}_context_seed_{context_seed}")
                
    elif args.function=="learn":
        for n,h,s,seed,c,context_seed,d in it.product(args.num_vars,args.num_hard,
                                         args.num_soft,args.model_seeds,
                                         args.num_context,args.context_seeds,args.data_size):
            learn_model(n,n,h,s,seed,c,context_seed,d)
    
    elif args.function=="evaluate":
        csvfile = open("results/evaluation"+ ".csv","w")
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(["num_vars","num_hard","num_soft","model_seed",
                             "num_context","context_seed","data_size","score",
                             "recall","precision","regret","time_taken"])
        for n,h,s,seed in it.product(args.num_vars,args.num_hard,
                                     args.num_soft,args.model_seeds):
            param = f"_n_{n}_max_clause_length_{n}_num_hard_{h}_num_soft_{s}_model_seed_{seed}"
            pickle_var = pickle.load(open("pickles/target_model" + param + ".pickle", "rb"))
            target_model = pickle_var["true_model"]
            for c,context_seed,d in it.product(args.num_context,
                                               args.context_seeds,args.data_size):
                tag = param+f"_num_context_{c}_num_data_{d}_context_seed_{context_seed}"
                pickle_var = pickle.load(open("pickles/learned_model" + tag + ".pickle", "rb"))
                learned_model = pickle_var["learned_model"]
                time_taken=pickle_var["time_taken"]
                score=pickle_var["score"]
                recall,precision,regret=evaluate_statistics(n,target_model,
                                                            learned_model,
                                                            args.sample_size,seed)
                print(n,h,s,d,score,recall,precision,regret)
                filewriter.writerow([n,h,s,seed,c,context_seed,d,score,
                                     recall,precision,regret,time_taken])
#        print(np.mean(scores),times_taken)
        csvfile.close()





