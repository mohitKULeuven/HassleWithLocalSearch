#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:46:25 2020

@author: mohit
"""

import numpy as np
import time
from typing import List
from .type_def import Clause
from .maxsat import MaxSAT
import copy
import os
import pickle
from tqdm import tqdm
import max_sat


# def eval_neighbours(
#     correct_examples, neighbours, data, labels, contexts, num_neighbours, rng, inf=None
# ):
#     if not inf:
#         inf = [None] * len(labels)
#
#     neighbours = copy.copy(neighbours)
#     next_correct_examples = np.zeros([len(neighbours), data.shape[0]])
#
#     scores = [0 for i in range(len(neighbours))]
#
#     for m, nbr in enumerate(neighbours):
#         optimums = {}
#         for i, example in enumerate(data):
#             key = "_".join(map(str, contexts[i]))
#             if key not in optimums:
#                 optimums[key] = nbr.optimal_value(contexts[i])
#             if correct_examples[i] == 1 and nbr.is_correct(
#                 example, labels[i], contexts[i], inf=inf[i], optimum=optimums[key]
#             ):
#                 next_correct_examples[m, i] = 1
#                 scores[m] += 1
#
#     lst_scores = []
#     lst_models = []
#     lst_correct_examples = []
#     for _ in range(num_neighbours):
#         lst_scores.append(max(scores))
#         best_index = rng.choice(
#             [i for i, v in enumerate(scores) if v == lst_scores[-1]]
#         )
#         lst_models.append(neighbours[best_index])
#         del scores[best_index]
#         del neighbours[best_index]
#         optimums = {}
#         for i, example in enumerate(data):
#             if correct_examples[i] == 0:
#                 key = "_".join(map(str, contexts[i]))
#                 if key not in optimums:
#                     optimums[key] = lst_models[-1].optimal_value(contexts[i])
#                 if lst_models[-1].is_correct(
#                     example, labels[i], contexts[i], inf=inf[i], optimum=optimums[key]
#                 ):
#                     next_correct_examples[best_index, i] = 1
#                     lst_scores[-1] += 1
#         lst_correct_examples.append(next_correct_examples[best_index, :])
#     return lst_models, lst_scores, lst_correct_examples


def eval_neighbours(neighbours, data, labels, contexts, num_neighbours, rng, inf=None):
    if not inf:
        inf = [None] * len(labels)

    neighbours = copy.copy(neighbours)
    next_correct_examples = np.zeros([len(neighbours), data.shape[0]])

    scores = [0 for i in range(len(neighbours))]

    for m, nbr in enumerate(neighbours):
        optimums = {}
        for i, example in enumerate(data):
            key = "_".join(map(str, contexts[i]))
            if key not in optimums:
                optimums[key] = nbr.optimal_value(contexts[i])
            if nbr.is_correct(
                example, labels[i], contexts[i], inf=inf[i], optimum=optimums[key]
            ):
                next_correct_examples[m, i] = 1
                scores[m] += 1

    lst_scores = []
    lst_models = []
    for _ in range(num_neighbours):
        lst_scores.append(max(scores))
        best_index = rng.choice(
            [i for i, v in enumerate(scores) if v == lst_scores[-1]]
        )
        lst_models.append(neighbours[best_index])
        del scores[best_index]
        del neighbours[best_index]
    return lst_models, lst_scores


def walk_sat(neighbours, data, labels, contexts, rng, inf=None):
    # prev_score = len(correct_examples)
    # if rng.random_sample() < p:
    #     next_model = neighbours[rng.randint(0, len(neighbours))]
    #     score, correct_examples = next_model.score(data, labels, contexts, inf)
    #     return next_model, score
    next_models, scores = eval_neighbours(
        neighbours, data, labels, contexts, 1, rng, inf
    )
    return next_models[0], scores[0]


def novelty(prev_model, neighbours, data, labels, contexts, rng, inf=None):
    # if rng.random_sample() < p:
    #     next_model = neighbours[rng.randint(0, len(neighbours))]
    #     score, correct_examples = next_model.score(data, labels, contexts, inf)
    #     return next_model, score
    lst_models, lst_scores = eval_neighbours(
        neighbours, data, labels, contexts, 2, rng, inf
    )
    if not lst_models[0].is_same(prev_model):
        return lst_models[0], lst_scores[0]
    else:
        return lst_models[1], lst_scores[1]


def novelty_plus(prev_model, neighbours, data, labels, contexts, wp, rng, inf=None):
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts, inf)
        return next_model, score
    return novelty(prev_model, neighbours, data, labels, contexts, rng, inf)


def adaptive_novelty_plus(
    prev_model,
    neighbours,
    data,
    labels,
    contexts,
    wp,
    theta,
    phi,
    best_scores,
    rng,
    inf=None,
):
    steps = int(len(labels) * theta)
    if len(best_scores) > steps:
        if best_scores[-steps] == best_scores[-1]:
            wp = wp + (1 - wp) * phi
        else:
            wp = wp - (wp * 2 * phi)
    if rng.random_sample() < wp:
        next_model = neighbours[rng.randint(0, len(neighbours))]
        score, correct_examples = next_model.score(data, labels, contexts, inf)
        return next_model, score, wp
    next_model, score = novelty(
        prev_model, neighbours, data, labels, contexts, rng, inf
    )
    return next_model, score, wp


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


def random_model(num_var, num_constraints, clause_len, seed) -> MaxSAT:
    rng = np.random.RandomState(seed)
    c = [rng.randint(0, 2) for i in range(num_constraints)]
    w = [float("%.3f" % (rng.uniform(0.001, 0.999))) for i in range(num_constraints)]
    l = []
    i = 1
    while i <= num_constraints:
        clause = []
        sample = rng.choice(num_var, clause_len, replace=False)
        for j in range(num_var):
            if j in sample:
                clause.append(int(rng.choice([-1, 0, 1])))
            else:
                clause.append(0)

        if clause not in l:
            l.append(clause)
            i += 1
    return MaxSAT(c, w, l)


def random_incorrect_example_index(model, data, contexts, labels, infeasible, rng):
    x = list(enumerate(np.copy(data)))
    rng.shuffle(x)
    indices, data = zip(*x)
    if not infeasible:
        infeasible = [None] * len(indices)
    index = 0
    for i, example in enumerate(data):
        if not model.is_correct(
            example, labels[indices[i]], contexts[indices[i]], infeasible[indices[i]]
        ):
            index = i
            break
    return indices[index]


def learn_weighted_max_sat(
    num_constraints: int,
    clause_len: int,
    data: np.ndarray,
    labels: np.ndarray,
    contexts: List[Clause],
    method,
    param,
    inf=None,
    p=0.01,
    wp=0.1,
    theta=0.17,
    phi=0.2,
    cutoff_time=5,
    seed=1,
    use_knowledge_compilation=False,
    observers=None
):
    """
    Learn a weighted MaxSAT model from examples. Contexts and clauses are set-encoded, i.e., they are represented by
    sets containing positive or negative integers in the range -n-1 to n+1. If a set contains an positive integer i, the i-1th
     Boolean feature is set to True, if it contains a negative integer -i, the i-1th Boolean feature is set to False.
    :param num_constraints:
        The number of clauses in the MaxSAT model to learn
    :param data:
        A Boolean s x n (number examples x number Boolean variables) numpy array in which every row is an example and
        every column a Boolean feature (True = 1 or False = 0)
    :param labels:
        A Boolean numpy array (length s) where the kth entry contains the label (1 or 0) of the kth example
    :param contexts:
        A list of s set-encoded contexts.
    :param use_knowledge_compilation:
        A Boolean flag denoting whether evaluation of a model's neighbours should use knowledge compilation
    :return:
        A list of weights and clauses. Every entry of the list is a tuple containing as first element None for hard
        constraints (clauses) or a floating point number for soft constraints, and as second element a set-encoded clause.
    """
    if not os.path.exists("pickles/learned_model"):
        os.makedirs("pickles/learned_model")

    # Initialising timers
    initialisation_time = 0
    random_restart_time = 0
    computing_neighbours_time = 0
    evaluation_time = 0
    cumulative_time = 0

    # Starting with a random model
    time_point = time.time()
    model = random_model(data.shape[1], num_constraints, clause_len, seed)
    initialisation_time += time.time() - time_point

    # Evaluating the initial model
    bar = tqdm("Score", total=100)
    time_point = time.time()
    if use_knowledge_compilation:
        examples = [[contexts[i], data[i], labels[i]] for i in range(len(data))]
        model_as_phenotype = max_sat.to_phenotype(max_sat.MaxSAT_to_genotype(model))
        score = int(max_sat.evaluate_knowledge_compilation_based(model_as_phenotype, examples)[0] * len(examples))
    else:
        score, correct_examples = model.score(data, labels, contexts, inf)
    evaluation_time += time.time() - time_point
    bar.update(score * 100 / data.shape[0])

    # Update cumulative time
    cumulative_time = initialisation_time + random_restart_time + computing_neighbours_time + evaluation_time
    
    # Some setup
    rng = np.random.RandomState(seed)
    prev_model = model
    solutions = [model.deep_copy().maxSatModel()]
    best_scores = [score]
    time_taken = [cumulative_time]
    for observer in observers:
        observer.observe_generation(
            gen_count=0,
            best_score=score/data.shape[0],
            gen_duration=time_taken[-1],
            cumulative_time=cumulative_time,
            initialisation_time=initialisation_time,
            random_restart_time=random_restart_time,
            computing_neighbours_time=computing_neighbours_time,
            evaluation_time=evaluation_time
        )
    iterations = [0]
    itr = 0
    num_neighbours = [0]
    nbr = 0
    num_example = data.shape[0]
    last_update_time = cumulative_time

    while (
        score < len(labels)
        and cumulative_time < cutoff_time
    ):
        if cumulative_time - last_update_time > cutoff_time / 4:
            # if rng.random_sample() < p:

            # Random restart
            time_point = time.time()
            next_model = random_model(data.shape[1], num_constraints, clause_len, seed)
            random_restart_time += time.time() - time_point

            time_point = time.time()
            if use_knowledge_compilation:
                model_as_phenotype = max_sat.to_phenotype(max_sat.MaxSAT_to_genotype(model))
                score = int(max_sat.evaluate_knowledge_compilation_based(model_as_phenotype, examples)[0]*len(examples))
            else:
                score, correct_examples = next_model.score(data, labels, contexts, inf)
            evaluation_time += time.time() - time_point

        else:
            # Compute neighbourhood
            time_point = time.time()
            if "naive" in param:
                neighbours = model.valid_neighbours()
            else:
                index = random_incorrect_example_index(
                    model, data, contexts, labels, inf, rng
                )
                if inf:
                    infeasible = inf
                else:
                    infeasible = [None] * len(data)
                neighbours = model.get_neighbours(
                    data[index],
                    contexts[index],
                    labels[index],
                    clause_len,
                    rng,
                    infeasible[index],
                )
            computing_neighbours_time += time.time() - time_point

            if len(neighbours) == 0 or (method != "walk_sat" and len(neighbours) < 2):
                cumulative_time = initialisation_time + random_restart_time + computing_neighbours_time + evaluation_time
                continue
            nbr += len(neighbours)

            # Compute model update
            time_point = time.time()
            if use_knowledge_compilation:
                # If evaluation should use knowledge compilation, walk_sat is automatically used at present
                next_model, score_as_proportion = max_sat.compute_best_neighbour_knowledge_compilation(neighbours, examples)
                score = round(score_as_proportion * len(examples))
            else:
                if method == "walk_sat":
                    next_model, score = walk_sat(
                        neighbours, data, labels, contexts, rng, inf
                    )
                elif method == "novelty":
                    next_model, score = novelty(
                        prev_model, neighbours, data, labels, contexts, rng, inf
                    )
                elif method == "novelty_plus":
                    next_model, score = novelty_plus(
                        prev_model, neighbours, data, labels, contexts, wp, rng, inf
                    )
                elif method == "adaptive_novelty_plus":
                    next_model, score, wp = adaptive_novelty_plus(
                        prev_model,
                        neighbours,
                        data,
                        labels,
                        contexts,
                        wp,
                        theta,
                        phi,
                        best_scores,
                        rng,
                        inf,
                    )
            # Computing a model update almost entirely comes down to evaluation all the model's neighbours, so
            # we include the time this update takes in the evaluation time
            evaluation_time += time.time() - time_point
            
        prev_model = model
        model = next_model
        itr += 1

        # Update cumulative time
        old_cumulative_time = cumulative_time
        cumulative_time = initialisation_time + random_restart_time + computing_neighbours_time + evaluation_time
        
        if score > best_scores[-1]:
            # Found a new best model
            solutions.append(model.deep_copy().maxSatModel())
            bar.update((score - best_scores[-1]) * 100 / num_example)
            iterations.append(itr)
            num_neighbours.append(nbr)
            best_scores.append(score)
            last_update_time = cumulative_time
            time_taken.append(cumulative_time)
        for observer in observers:
            observer.observe_generation(
                itr,
                best_scores[-1] / data.shape[0],
                gen_duration = cumulative_time - old_cumulative_time,
                cumulative_time=cumulative_time,
                initialisation_time=initialisation_time,
                random_restart_time=random_restart_time,
                computing_neighbours_time=computing_neighbours_time,
                evaluation_time=evaluation_time
            )

    for i, score in enumerate(best_scores):
        best_scores[i] = score * 100 / num_example
    pickle_var = {
        "learned_model": solutions,
        "time_taken": time_taken,
        "score": best_scores,
        "iterations": iterations,
        "num_neighbour": num_neighbours,
    }
    if "cnf" in param:
        pickle_var = {
            "learned_model": [solutions[-1]],
            "time_taken": [time_taken[-1]],
            "score": [best_scores[-1]],
            "iterations": [iterations[-1]],
            "num_neighbour": num_neighbours,
        }
    pickle.dump(pickle_var, open("pickles/learned_model/" + param + ".pickle", "wb"))
    print(f"Iterations: {itr}")
    print(f"Timing:\n"
          f"Initialisation: {initialisation_time}\n"
          f"Random restarts: {random_restart_time}\n"
          f"Computing neighbours: {computing_neighbours_time}\n"
          f"Evaluation: {evaluation_time}\n"
          f"Total: {cumulative_time}\n")
    return solutions[-1]
    # return (solutions, best_scores, time_taken, iterations, num_neighbours)
