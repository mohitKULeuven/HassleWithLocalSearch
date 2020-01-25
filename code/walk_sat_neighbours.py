#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 10:44:15 2020

@author: mohit
"""
import MaxSAT
from pysat_solver import solve_weighted_max_sat
import numpy as np

def hc_sat_ex(maxsat:MaxSAT,instance,context,rng):
    model=maxsat.maxSatModel()
    lst=[]
    i=0
    for w,clause in model:
        covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        if w is None and covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)
            

def hc_not_sat_ex(maxsat:MaxSAT,instance,context,rng):
    model=maxsat.maxSatModel()
    lst=[]
    i=0
    for w,clause in model:
        covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        if w is None and not covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)

def sc_sat_ex_not_opt(model:MaxSAT,instance,context,rng):
    opt,cost=solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1)
    lst=[]
    i=0
    for w,clause in model.maxSatModel():
        ex_covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        opt_covered = any(not opt[abs(i) - 1] if i < 0 
                                   else opt[i - 1] for i in clause)
        if w is not None and ex_covered and not opt_covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)
    

def sc_sat_opt_not_ex(model:MaxSAT,instance,context,rng):
    opt,cost=solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1)
    lst=[]
    i=0
    for w,clause in model.maxSatModel():
        ex_covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        opt_covered = any(not opt[abs(i) - 1] if i < 0 
                                   else opt[i - 1] for i in clause)
        if w is not None and opt_covered and not ex_covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)

def sc_not_sat_any(model:MaxSAT,instance,context,rng):
    opt,cost=solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1)
    lst=[]
    i=0
    for w,clause in model.maxSatModel():
        ex_covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        opt_covered = any(not opt[abs(i) - 1] if i < 0 
                                   else opt[i - 1] for i in clause)
        if w is not None and not opt_covered and not ex_covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)

def sc_sat_both(model:MaxSAT,instance,context,rng):
    opt,cost=solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1)
    lst=[]
    i=0
    for w,clause in model.maxSatModel():
        ex_covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        opt_covered = any(not opt[abs(i) - 1] if i < 0 
                                   else opt[i - 1] for i in clause)
        if w is not None and opt_covered and ex_covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)

def sc_not_sat_ex(maxsat:MaxSAT,instance,context,rng):
    model=maxsat.maxSatModel()
    lst=[]
    i=0
    for w,clause in model:
        ex_covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        if w is not None and not ex_covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)

def sc_sat_ex(maxsat:MaxSAT,instance,context,rng):
    model=maxsat.maxSatModel()
    lst=[]
    i=0
    for w,clause in model:
        ex_covered = any(not instance[abs(i) - 1] if i < 0 
                                   else instance[i - 1] for i in clause)
        if w is not None and ex_covered:
            lst.append(i)
        i+=1
    if not lst:
        return -1
    return rng.choice(lst)

def remove_literal(model,clause_index,literals):
    neighbours=[]
    for literal in literals:
        j=abs(literal)-1
        if model.l[clause_index][j]==np.sign(literal):
            neighbour=model.deep_copy()
            neighbour.l[clause_index][j]=0
            if any(neighbour.l[clause_index]):
                neighbours.append(neighbour)
    return neighbours

def add_literal(model,clause_index,literals):
    neighbours=[]
    for literal in literals:
        j=abs(literal)-1
        if model.l[clause_index][j]==0:
            neighbour=model.deep_copy()
            neighbour.l[clause_index][j]=int(np.sign(literal))
            if any(neighbour.l[clause_index]):
                neighbours.append(neighbour)
    return neighbours

def instance_to_literals(instance):
    literals=set()
    for i,elem in enumerate(instance):
        if elem:
            literals.add(i+1)
        else:
            literals.add(-i-1)
    return literals

def neighbours_inf(model:MaxSAT,instance,context,rng):
    i=hc_not_sat_ex(model,instance,context,rng)
    neighbours=[]
    neighbour=model.deep_copy()
    neighbour.c[i]=1-neighbour.c[i]
    neighbours.append(neighbour)
    for j in range(model.n):
        values=[-1,0,1]
        values.remove(model.l[i][j])
        for val in values:
            neighbour=model.deep_copy()
            neighbour.l[i][j]=val
            if any(neighbour.l[i]):
                neighbours.append(neighbour)
        
    return neighbours

def neighbours_sub(model:MaxSAT,instance,context,rng):
    sol,cost=solve_weighted_max_sat(model.n, model.maxSatModel(), context, 1)
    opt_literals=instance_to_literals(sol)
    exp_literals=instance_to_literals(instance)
    neighbours=[]
    
    index=hc_sat_ex(model,instance,context,rng)
    if index>=0:
        neighbours.extend(remove_literal(model,index,opt_literals))
    
    index=sc_sat_ex_not_opt(model,instance,context,rng)
    if index>=0:
        neighbour=model.deep_copy()
        neighbour.c[index]=1-neighbour.c[index]
        neighbours.append(neighbour)
    
    index=sc_sat_opt_not_ex(model,instance,context,rng)
    if index>=0:
        neighbours.extend(remove_literal(model,index,opt_literals))
        neighbours.extend(add_literal(model,index,exp_literals))
    
    index=sc_not_sat_any(model,instance,context,rng)
    if index>=0:
        neighbours.extend(add_literal(model,index,exp_literals-opt_literals))
    
    index=sc_sat_both(model,instance,context,rng)
    if index>=0:
        neighbours.extend(remove_literal(model,index,opt_literals-exp_literals))
        
    return neighbours

def neighbours_pos(model:MaxSAT,instance,context,rng):
    exp_literals=instance_to_literals(instance)
    
    neighbours=[]
    
    index=hc_sat_ex(model,instance,context,rng)
    if index>=0:
        neighbours.extend(remove_literal(model,index,exp_literals))
    
    index=sc_not_sat_ex(model,instance,context,rng)
    if index>=0:
        neighbour=model.deep_copy()
        neighbour.c[index]=1-neighbour.c[index]
        neighbours.append(neighbour)
    
    index=sc_sat_ex(model,instance,context,rng)
    if index>=0:
        neighbours.extend(remove_literal(model,index,exp_literals))
        
    return neighbours
    
    
    
    
    
    
    
    
    
    
    
    
    