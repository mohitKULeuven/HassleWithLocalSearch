#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:36:56 2020

@author: mohit
"""
import numpy as np
from .local_search import learn_weighted_max_sat

from .pysat_solver import solve_weighted_max_sat
from .experiments.synthetic import evaluate_statistics


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


if __name__ == "__main__":
    example3()
