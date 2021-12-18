#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:36:56 2020

@author: mohit
"""
import numpy as np
from .local_search import learn_weighted_max_sat

from .pysat_solver import solve_weighted_max_sat, get_value
from .experiments.synthetic import evaluate_statistics
# from hassle.pysat_solver import solve_weighted_max_sat, get_value, label_instance
import pickle

def example1():
    # Example
    #      A \/ B
    # 1.0: A
    #
    # --:  A, B   sat  opt (1)
    # --:  A,!B   sat  opt (1)
    # --: !A, B   sat !opt (0)
    # --: !A,!B  !sat !opt (0)
    #
    #  A:  A, B   sat  opt (1)
    #  A:  A,!B   sat  opt (1)
    #
    # !A: !A, B   sat  opt (0)
    # !A: !A,!B  !sat  opt (0)
    #
    #  B:  A, B   sat  opt (1)
    #  B: !A, B   sat !opt (0)
    #
    # !B:  A,!B   sat  opt (1)
    # !B: !A,!B  !sat !opt (0)

    data = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [True, True],
            [False, True],
            [True, False],
            [False, False],
        ]
    )

    labels = np.array(
        [True, True, False, False, True, True, True, False, True, False, True, False]
    )
    contexts = [set(), set(), set(), set(), {1}, {1}, {-1}, {-1}, {2}, {2}, {-2}, {-2}]
    # learn_weighted_max_sat(2,2, data, labels, contexts, 12, 0.1, 5, 1)


def example2():
    # Example
    #      !A \/ !B \/ !C
    # 1.0: A
    # 0.5: B \/ !C
    #
    # pos  A, B,!C  A
    # neg  A,!B, C  A suboptimal
    # neg  A, B, C  A infeasible
    #
    # pos !A, B,!C !A
    # neg !A,!B, C !A suboptimal
    #
    # pos  A, B,!C  B
    # neg !A, B, C  B suboptimal
    # neg  A, B, C  B infeasible
    #
    # pos  A,!B,!C !B
    # neg !A,!B, C !B suboptimal
    #
    # pos  A,!B, C  C
    # neg !A, B, C  C suboptimal
    # neg  A, B, C  C infeasible
    #
    # pos  A, B,!C !C
    # neg !A,!B,!C !C suboptimal
    #
    # pos !A,!B,!C  !A,!B
    # pos  A,!B,!C  !B,!C
    # pos  !A,B,C  B,C

    data = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
            [True, False, False],
            [False, False, True],
            [True, False, True],
            [False, True, True],
            [True, True, True],
            [True, True, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, True],
        ]
    )

    labels = np.array(
        [
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
        ]
    )

    contexts = [
        {1},
        {1},
        {1},
        {-1},
        {-1},
        {2},
        {2},
        {2},
        {-2},
        {-2},
        {3},
        {3},
        {3},
        {-3},
        {-3},
        {-1, -2},
        {-2, -3},
        {2, 3},
    ]

def example3():
    n=3
    clause1 = (1,2,3)
    clause2 = (1,)
    clause3 = (2,-3)
    model1=[(None, clause1),(1, clause2),(1, clause3)]
    context=(-1,-2)
    sol=solve_weighted_max_sat(3,model1,context,5)
    assert len(sol[0])==1

    data=sol[0]
    labels=[True]*len(sol[0])
    contexts = [{-1, -2}] * len(labels)

    data.append([False, False, False])
    labels.append(False)
    contexts.append(set())
    data.append([True, True, False])
    labels.append(True)
    contexts.append(set())
    data.append([True, True, True])
    labels.append(True)
    contexts.append({3})
    data.append([True, True, False])
    labels.append(False)
    contexts.append({3})

    data=np.array(data)
    labels=np.array(labels)


    lmodel1 = learn_weighted_max_sat(3, 3, data, labels, contexts, "walk_sat", "")

    contexts = [set()] * len(labels)
    lmodel2 = learn_weighted_max_sat(3, 3, data, labels, contexts, "walk_sat", "")
    print(data, labels)
    print(lmodel1)
    print(lmodel2)

    recall, precision, accuracy, reg, infeasiblity=evaluate_statistics(
        n, model1, lmodel1, None
    )
    print(recall, precision, accuracy, reg, infeasiblity)

    recall, precision, accuracy, reg, infeasiblity = evaluate_statistics(
        n, model1, lmodel2, None
    )
    print(recall, precision, accuracy, reg, infeasiblity)


def context_relevance(n, h, s, seed, c, num_pos, num_neg, neg_type, context_seed):
    param = f"_n_{n}_max_clause_length_{int(n / 2)}_num_hard_{h}_num_soft_{s}_model_seed_{seed}"
    target_model = pickle.load(
        open("pickles/target_model/" + param + ".pickle", "rb")
    )["true_model"]
    tag_cnd = (
            param
            + f"_num_context_{c}_num_pos_{num_pos}_num_neg_{num_neg}_context_seed_{context_seed}"
    )
    if neg_type:
        tag_cnd = (
                param
                + f"_num_context_{c}_num_pos_{num_pos}_num_neg_{num_neg}_neg_type_{neg_type}_context_seed_{context_seed}"
        )
    pickle_cnd = pickle.load(
        open("pickles/contexts_and_data/" + tag_cnd + ".pickle", "rb")
    )
    sol, cost = solve_weighted_max_sat(n, target_model, None, 1)
    opt_val = get_value(target_model, sol, None)
    count=0
    # print(target_model)
    # print(pickle_cnd["data"], pickle_cnd["labels"])
    for c in pickle_cnd["contexts"]:
        # print(c)
        sol, cost = solve_weighted_max_sat(n, target_model, c, 1)
        # print(sol,cost)
        # print(target_model, sol, c)
        if opt_val == get_value(target_model, sol, c):
            count+=1
    return count, len(pickle_cnd["contexts"])


if __name__ == "__main__":
    print(context_relevance(5,2,2,111,25,2,2,"both",111))
    # example3()
