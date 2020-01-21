#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:46:25 2020

@author: mohit
"""

import numpy as np
import time
from typing import List
from type_def import MaxSatModel, Clause, suppress_stdout, Instance, Context
import MaxSAT
import itertools as it

def learn_weighted_max_sat(
    m: int, data: np.ndarray, labels: np.ndarray, contexts: List[Clause], 
    cutoff_score:int, cutoff_time:int, seed: int
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

    rng = np.random.RandomState(seed)
    c=[rng.randint(0,2) for i in range(m)]
    w=[1 for i in range(m)]
    random_clauses=[  rng.randint(1,pow(3,data.shape[1])) for i in range(m)  ]
    l=[ ternary(i,data.shape[1]) for i in random_clauses ]
    l=[ [-1 if j==2 else j for j in clause] for clause in l ]
    
#    l=[[rng.choice([-1,0,1]) for j in range(data.shape[1])] for i in range(m)]
    model = MaxSAT.MaxSAT(c,w,l)
    
    score,correct_examples=model.score(data,labels,contexts)
    print(model.maxSatModel(),score)
    solution=model.deep_copy()
    best_score=score
    time_taken=time.time()-start
    
    while score<cutoff_score and time.time()-start<cutoff_time:
        neighbours=model.valid_neighbours()
        model,score,correct_examples=best_neighbour(model,correct_examples,
                                                    neighbours,data,labels,
                                                    contexts,rng)
        print(model.maxSatModel(),score)
        if score > best_score:
            solution=model.deep_copy()
            best_score=score
            time_taken=time.time()-start
#        break
    print(f"time taken: {time_taken} seconds")   
    print(solution.maxSatModel(),best_score)     
    return solution.maxSatModel()
    
 
def best_neighbour(model,correct_examples,neighbours,data,labels,contexts,rng):
    """
    Returns a model which breaks least of the 
    already satisfied examples with it's score
    """
#    c=[0,1]
#    w=[1,1]
#    l=[[1,0],[1,1]]
#    model = MaxSAT.MaxSAT(c,w,l)
#    neighbours=[model]
    next_correct_examples=np.zeros([len(neighbours),data.shape[0]])
    
    scores=[0 for i in range(len(neighbours))]
    for m,next_model in enumerate(neighbours):
        for i,example in enumerate(data):   
#            print(i,correct_examples[i],next_model.is_correct(example,labels[i],contexts[i]))
            if ( correct_examples[i]==1 and 
                next_model.is_correct(example,labels[i],contexts[i])
                ):
                next_correct_examples[m,i]=1
                scores[m]+=1
    
    best_score=max(scores)
#    print(best_score)
#    best_index=scores.index(best_score)
    best_index=rng.choice([i for i,v in enumerate(scores) if v==best_score])
    best_model=neighbours[best_index]
    for i,example in enumerate(data):
        if ( correct_examples[i]==0 and 
            best_model.is_correct(example,labels[i],contexts[i])
            ):
            next_correct_examples[best_index,i]=1
            best_score+=1           
    return best_model,best_score,next_correct_examples[best_index,:]
        
  
def ternary(n,length):
    e=n//3
    q=n%3
    if length>1:
        if n==0:
            return ternary(e,length-1)+[0]
        elif e==0:
            return ternary(e,length-1)+[q]
        else:
            return ternary(e,length-1) + [q]
    else:
        if n==0:
            return [0]
        elif e==0:
            return [q]
        else:
            return ternary(e,length-1) + [q]
        


"""
Next part of the code is for testing purpose only
"""
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
    learn_weighted_max_sat(2, data, labels, contexts,12,10,1)


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

    learn_weighted_max_sat(3, data, labels, contexts,18,200,1)


if __name__ == "__main__":
    example2()







