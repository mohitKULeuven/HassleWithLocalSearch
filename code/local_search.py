#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:46:25 2020

@author: mohit
"""

import numpy as np
import time
from typing import List
from type_def import MaxSatModel, Clause
import MaxSAT
import matplotlib.pyplot as plt
import copy


def eval_neighbours(
    model, correct_examples, neighbours, data, labels, contexts, num_neighbours, rng
):
    neighbours = copy.copy(neighbours)
    next_correct_examples = np.zeros([len(neighbours), data.shape[0]])

    scores = [0 for i in range(len(neighbours))]
    for m, next_model in enumerate(neighbours):
        for i, example in enumerate(data):
            if correct_examples[i] == 1 and next_model.is_correct(
                example, labels[i], contexts[i]
            ):
                next_correct_examples[m, i] = 1
                scores[m] += 1

    lst_scores = []
    lst_models = []
    lst_correct_examples = []
    for _ in range(num_neighbours):

        lst_scores.append(max(scores))
        best_index = rng.choice(
            [i for i, v in enumerate(scores) if v == lst_scores[-1]]
        )
        lst_models.append(neighbours[best_index])
        del scores[best_index]
        del neighbours[best_index]

        for i, example in enumerate(data):
            if correct_examples[i] == 0 and lst_models[-1].is_correct(
                example, labels[i], contexts[i]
            ):
                next_correct_examples[best_index, i] = 1
                lst_scores[-1] += 1
        lst_correct_examples.append(next_correct_examples[best_index, :])
    return lst_models, lst_scores, lst_correct_examples


def walk_sat(model, correct_examples, neighbours, data, labels, contexts, p, rng):
    prev_score = len(correct_examples)
    lst_models, lst_scores, lst_correct_examples = eval_neighbours(
        model, correct_examples, neighbours, data, labels, contexts, 1, rng
    )
    next_model, score, correct_examples = (
        lst_models[0],
        lst_scores[0],
        lst_correct_examples[0],
    )
    if score == prev_score:
        return next_model, score, correct_examples
    elif rng.random_sample() < p:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts)
        return next_model, score, correct_examples
    else:
        return next_model, score, correct_examples


def novelty(
    model, prev_model, correct_examples, neighbours, data, labels, contexts, p, rng
):
    lst_models, lst_scores, lst_correct_examples = eval_neighbours(
        model, correct_examples, neighbours, data, labels, contexts, 2, rng
    )
    if not lst_models[0].is_same(prev_model):
        return lst_models[0], lst_scores[0], lst_correct_examples[0]
    elif rng.random_sample() > p:
        return lst_models[0], lst_scores[0], lst_correct_examples[0]
    else:
        return lst_models[1], lst_scores[1], lst_correct_examples[1]


def novelty_plus(
    model, prev_model, correct_examples, neighbours, data, labels, contexts, p, wp, rng
):
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts)
        return next_model, score, correct_examples
    return novelty(
        model, prev_model, correct_examples, neighbours, data, labels, contexts, p, rng
    )


def adaptive_novelty_plus(
    model,
    prev_model,
    correct_examples,
    neighbours,
    data,
    labels,
    contexts,
    p,
    wp,
    theta,
    phi,
    best_scores,
    rng,
):
    steps = int(len(labels) * theta)
    if len(best_scores) > steps:
        if best_scores[-steps] == best_scores[-1]:
            wp = wp + (1 - wp) * phi
        else:
            wp = wp - (wp * 2 * phi)
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts)
        return next_model, score, correct_examples, wp
    next_model, score, correct_examples = novelty(
        model, prev_model, correct_examples, neighbours, data, labels, contexts, p, rng
    )
    return next_model, score, correct_examples, wp


def best_neighbour(
    model,
    prev_model,
    correct_examples,
    neighbours,
    data,
    labels,
    contexts,
    method,
    p,
    wp,
    theta,
    phi,
    best_scores,
    rng,
):
    """
    Returns a model which breaks least of the 
    already satisfied examples with it's score
    """
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts)
        if method == "adaptive_novelty_plus":
            return next_model, score, correct_examples, wp
        return next_model, score, correct_examples
    else:
        if method == "walk_sat":
            return walk_sat(
                model, correct_examples, neighbours, data, labels, contexts, p, rng
            )

        elif method == "novelty":
            return novelty(
                model,
                prev_model,
                correct_examples,
                neighbours,
                data,
                labels,
                contexts,
                p,
                rng,
            )

        elif method == "novelty_plus":
            return novelty_plus(
                model,
                prev_model,
                correct_examples,
                neighbours,
                data,
                labels,
                contexts,
                p,
                wp,
                rng,
            )

        else:
            return adaptive_novelty_plus(
                model,
                prev_model,
                correct_examples,
                neighbours,
                data,
                labels,
                contexts,
                p,
                wp,
                theta,
                phi,
                best_scores,
                rng,
            )


def ternary(n, length):
    e = n // 3
    q = n % 3
    if length > 1:
        if n == 0:
            return ternary(e, length - 1) + [0]
        elif e == 0:
            return ternary(e, length - 1) + [q]
        else:
            return ternary(e, length - 1) + [q]
    else:
        if n == 0:
            return [0]
        elif e == 0:
            return [q]
        else:
            return ternary(e, length - 1) + [q]


def learn_weighted_max_sat(
    m: int,data: np.ndarray,labels: np.ndarray,contexts: List[Clause],
    method,cutoff_score: int, w,
    p=0.1,wp=0.1,theta=0.17,phi=0.2,cutoff_time=5,seed=1
) -> MaxSatModel:
    """
    Learn a weighted MaxSAT model from examples. Contexts and clauses are set-encoded, i.e., they are represented by
    sets containing positive or negative integers in the range -n-1 to n+1. If a set contains an positive integer i, the i-1th
     Boolean feature is set to True, if it contains a negative integer -i, the i-1th Boolean feature is set to False.
    :param m:
        The number of clauses in the MaxSAT model to learn
    :param data:
        A Boolean s x n (number examples x number Boolean variables) numpy array in which every row is an example and
        every column a Boolean feature (True = 1 or False = 0)
    :param labels:
        A Boolean numpy array (length s) where the kth entry contains the label (1 or 0) of the kth example
    :param contexts:
        A list of s set-encoded contexts.
    :return:
        A list of weights and clauses. Every entry of the list is a tuple containing as first element None for hard
        constraints (clauses) or a floating point number for soft constraints, and as second element a set-encoded clause.
    """
    start = time.time()
    # starting with a random model
    scores = []
    best_scores = []
    rng = np.random.RandomState(seed)
    c = [rng.randint(0, 2) for i in range(m)]
    w = [1 for i in range(m)]
    
    l=[]
    i=1
#    print(data.shape)
    while i <= m:
        clause=[]
        for _ in range(data.shape[1]):
            clause.append(int(rng.choice([-1,0,1])))
        if clause not in l:
            l.append(clause)
            i+=1
#    random_clauses = [rng.randint(1, pow(3, data.shape[1])) for i in range(m)]
#    l = [ternary(i, data.shape[1]) for i in random_clauses]
#    l = [[-1 if j == 2 else j for j in clause] for clause in l]

    #    l=[[rng.choice([-1,0,1]) for j in range(data.shape[1])] for i in range(m)]
    model = MaxSAT.MaxSAT(c, w, l)
    prev_model = model

    score, correct_examples = model.score(data, labels, contexts)
    scores.append(score)
    print("Initial Score: ", score * 100 / data.shape[0])
    solution = model.deep_copy()
    best_score = score
    time_taken = time.time() - start
    iterations=0
    while score < cutoff_score and time.time() - start < cutoff_time:
#    while score < cutoff_score and iterations < cutoff_time:
        neighbours = model.walk_sat_neighbours(data, labels, contexts, rng,w)
        if len(neighbours) == 0:
            continue
        elif method != "walk_sat" and len(neighbours) < 2:
            continue

        if method == "adaptive_novelty_plus":
            next_model, score, correct_examples, wp = best_neighbour(
                model,
                prev_model,
                correct_examples,
                neighbours,
                data,
                labels,
                contexts,
                method,
                p,
                wp,
                theta,
                phi,
                best_scores,
                rng,
            )
        else:
            next_model, score, correct_examples = best_neighbour(
                model,
                prev_model,
                correct_examples,
                neighbours,
                data,
                labels,
                contexts,
                method,
                p,
                wp,
                theta,
                phi,
                best_scores,
                rng,
            )
        scores.append(score)
        prev_model = model
        model = next_model
        #        print(model.maxSatModel(),score)
        if score > best_score:
            solution = model.deep_copy()
            best_score = score
            time_taken = time.time() - start
        best_scores.append(best_score)
        iterations+=1
    #        break

    #    print(f"time taken: {time_taken} seconds")
    score_percentage = best_score * 100 / data.shape[0]
    print("Final Score: ", score_percentage)
    #    print(solution.maxSatModel(),best_score,best_score*100/data.shape[0])

    return solution.maxSatModel(), score_percentage, time_taken, scores, best_scores,iterations

