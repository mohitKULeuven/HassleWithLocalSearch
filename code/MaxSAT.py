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

class MaxSAT:
    
    def __init__(self,c,w,l):
        self.c=c #tells wether a constraint is hard
        self.w=w #tells wether a constraint is soft, always 1?
        self.l=l #tells wether a literal j is present in clause i
        self.k= len(l) #number of clauses
        self.n=len(l[0]) #number of variables
#        self.clauses=self.get_clauses(l)
        
    """
    when looking for a neighbour make sure that the 
    clauses are compatible with each other??
    """
    def valid_neighbours(self):
        neighbours=[]
        for i in range(self.k):
            neighbour=self.deep_copy()
            neighbour.c[i]=1-neighbour.c[i]
            neighbours.append(neighbour)
            for j in range(self.n):
                values=[-1,0,1]
                values.remove(self.l[i][j])
                for val in values:
                    neighbour=self.deep_copy()
                    neighbour.l[i][j]=val
                    if any(neighbour.l[i]):
                        neighbours.append(neighbour)
        
        return neighbours
    
    def random_neighbour(self,rng):
        neighbour=self.deep_copy()
#        np.random.seed(seed)
        random_clause=rng.randint(0,neighbour.k)
        random_vector=rng.randint(0,2)
        if random_vector==0:
            neighbour.c[random_clause]=1-neighbour.c[random_clause]
            return neighbour
        random_literal=rng.randint(0,neighbour.n)
        values=[-1,0,1]
        values.remove(neighbour.l[random_clause][random_literal])
        if len([i for i in neighbour.l[random_clause] if i!=0])==1 and neighbour.l[random_clause][random_literal]!=0:
            values.remove(0)
        neighbour.l[random_clause][random_literal]=int(rng.choice(values))
        return neighbour
    
    def score(self,data,labels,contexts):
        score=0
        correct_examples=[0]*data.shape[0]
        for i,example in enumerate(data):
            optimum=self.optimal_value(contexts[i])
            val=self.evaluate(example,contexts[i])
            if (labels[i]==0) and (val==None or val<optimum):
                score+=1
                correct_examples[i]=1
            elif (labels[i]==1) and (val!=None and val==optimum):
                score+=1
                correct_examples[i]=1
        return score,correct_examples
    
    def evaluate(self, example, context=[]):
        model=self.maxSatModel()
        if context:
            model.append((None,context))
        return get_value(model,example)
    
    def is_correct(self, example,label,context=[]):
        model=self.deep_copy().maxSatModel()
        if context:
            model.append((None,context))
        val = get_value(model,example)
        if val is None:
            return (not label)
        else:
            optimum=self.optimal_value(context)
            if val==optimum and label:
                return True
            if val<optimum and not label:
                return True
        return False
            
    
    def maxSatModel(self)->MaxSatModel:
        model=[]
        clauses=self.get_clauses(self.l)
        for i in range(self.k):
            if self.c[i]==1:
                model.append( (None, clauses[i]) )
            else:
                model.append( (self.w[i], clauses[i]) )
        return model

    def get_clauses(self,l=[]):
        if not l:
            l=self.l
        clauses=[]
        for i,constraint in enumerate(l):
            clause=[]
            for j,literal in enumerate(l[i]):
                if literal!=0:
                    clause.append((j+1)*literal)
            if clause:
                clauses.append(set(clause))
        return clauses
    
    def optimal_value(self, context=[]):
        sol,cost=solve_weighted_max_sat(self.n, self.maxSatModel(), context, 1)
        if not sol:
            return None
        model=self.deep_copy().maxSatModel()
        if context:
            model.append((None,context))
        return get_value(model,sol)
#        if sol:
#            return self.evaluate(sol[0],context)
#        return 0
    
    def deep_copy(self):
        c=copy.copy(self.c)
        w=copy.copy(self.w)
        l=[copy.copy(items) for items in self.l] 
        return MaxSAT(c,w,l)
                    
    def print_model(self):
        print(self.maxSatModel())
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    