#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:09:50 2020

@author: mohit
"""

import numpy as np
import copy
from type_def import MaxSatModel
from pysat_solver import solve_weighted_max_sat, get_value
from walk_sat_neighbours import neighbours_inf, neighbours_pos, neighbours_sub


class MaxSAT:
    def __init__(self, c=[], w=[], l=[[]]):
        self.c = c  # tells wether a constraint is hard
        self.w = w  # tells wether a constraint is soft, always 1?
        self.l = l  # tells wether a literal j is present in clause i
        self.k = len(l)  # number of clauses
        self.n = len(l[0])  # number of variables

    #        self.clauses=self.get_clauses(l)

    def from_max_sat_model(self, n: int, model: MaxSatModel):
        k = len(model)
        c = [0] * k
        w = [1] * k
        l = [[0] * n for i in range(k)]
        i = 0
        for weight, clause in model:
            if w is None:
                c[i] = 1
            else:
                w[i] = weight

            for literal in clause:
                l[i][np.abs(literal)] = np.sign(literal)
            i += 1
        self.c = c
        self.w = w
        self.l = l
        self.k = k
        self.n = n

    def walk_sat_neighbours(self, data, labels, contexts, rng,w):
        neighbours = []
        violated_examples = []
        index = 0
        for i, example in enumerate(data):
            if not self.is_correct(example, labels[i], contexts[i]):
                violated_examples.append(i)
        index = rng.choice(violated_examples)
        picked_example = data[index]
        val = get_value(self.maxSatModel(), picked_example, contexts[index])
        if not labels[index]:
            neighbours = neighbours_pos(self, picked_example, contexts[index], rng,w)
        elif val is None:
            neighbours = neighbours_inf(self, picked_example, contexts[index], rng)
        else:
            neighbours = neighbours_sub(self, picked_example, contexts[index], rng,w)

        return neighbours

    """
    when looking for a neighbour make sure that the 
    clauses are compatible with each other??
    """

    def valid_neighbours(self):
        neighbours = []
        for i in range(self.k):
            neighbour = self.deep_copy()
            neighbour.c[i] = 1 - neighbour.c[i]
            neighbours.append(neighbour)
            for j in range(self.n):
                values = [-1, 0, 1]
                values.remove(self.l[i][j])
                for val in values:
                    neighbour = self.deep_copy()
                    neighbour.l[i][j] = val
                    if any(neighbour.l[i]):
                        neighbours.append(neighbour)

        return neighbours

    def random_neighbour(self, rng):
        neighbour = self.deep_copy()
        #        np.random.seed(seed)
        random_clause = rng.randint(0, neighbour.k)
        random_vector = rng.randint(0, 2)
        if random_vector == 0:
            neighbour.c[random_clause] = 1 - neighbour.c[random_clause]
            return neighbour
        random_literal = rng.randint(0, neighbour.n)
        values = [-1, 0, 1]
        values.remove(neighbour.l[random_clause][random_literal])
        if (
            len([i for i in neighbour.l[random_clause] if i != 0]) == 1
            and neighbour.l[random_clause][random_literal] != 0
        ):
            values.remove(0)
        neighbour.l[random_clause][random_literal] = int(rng.choice(values))
        return neighbour

    def score(self, data, labels, contexts):
        """
        Number of correctly classified examples by the model
        """
        score = 0
        correct_examples = [0] * data.shape[0]
        for i, example in enumerate(data):
            if self.is_correct(example, labels[i], contexts[i]):
                score += 1
                correct_examples[i] = 1
        return score, correct_examples

    def is_correct(self, example, label, context=[]):
        val = get_value(self.maxSatModel(), example, context)
        if val is None:
            return not label
        else:
            optimum = self.optimal_value(context)
            if val == optimum and label:
                return True
            if val < optimum and not label:
                return True
        return False

    def maxSatModel(self) -> MaxSatModel:
        model = []
        clauses = self.get_clauses(self.l)
        for i in range(self.k):
            if self.c[i] == 1:
                model.append((None, clauses[i]))
            else:
                model.append((self.w[i], clauses[i]))
        return model

    def get_clauses(self, l=[]):
        if not l:
            l = self.l
        clauses = []
        for i, constraint in enumerate(l):
            clause = []
            for j, literal in enumerate(l[i]):
                if literal != 0:
                    clause.append((j + 1) * literal)
            if clause:
                clauses.append(set(clause))
        return clauses

    def optimal_value(self, context=[]):
        sol, cost = solve_weighted_max_sat(self.n, self.maxSatModel(), context, 1)
        if not sol:
            return None
        model = self.deep_copy().maxSatModel()
        if context:
            model.append((None, context))
        return get_value(model, sol)

    #        if sol:
    #            return self.evaluate(sol[0],context)
    #        return 0

    def is_same(self, model):
        c1, w1, l1 = zip(*sorted(zip(self.c, self.w, self.l)))
        c2, w2, l2 = zip(*sorted(zip(model.c, model.w, model.l)))
        if c1 == c2 and w1 == w2 and l1 == l2:
            return True
        return False

    def deep_copy(self):
        c = copy.copy(self.c)
        w = copy.copy(self.w)
        l = [copy.copy(items) for items in self.l]
        return MaxSAT(c, w, l)

    def print_model(self):
        print(self.maxSatModel())

    def model_to_string(self):
        model = self.maxSatModel()
        letters = "abcdefghijklmnopqrstuvwxyz".upper()

        if model is None:
            return "No model."

        def char(_l):
            if _l < 0:
                return f"!{letters[-_l - 1]}"
            else:
                return letters[_l - 1]

        result = ""
        for weight, clause in model:
            clause = " \\/ ".join(
                char(l)
                for l in sorted(clause, key=lambda x: (abs(x), 0 if x > 0 else 1))
            )
            result += (
                f"{'hard' if weight is None else 'soft'}, "
                f"{0.0 if weight is None else weight}: {clause}\n"
            )
        return result
