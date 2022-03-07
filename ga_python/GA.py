# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:29:54 2022

@author: mankleh
"""

import numpy as np
import time
from math import exp 

def Levy3(x):
    n = len(x)
    y = 1 + (x - 1) / 4
    # calculate f(y(x))
    term1 = np.sin(np.pi*y[0])**2
    term3 = (y[n-1]-1)**2 *(1+ np.sin(2*np.pi*y[n-1]))**2
    
    sum = 0 
    for x_i in y:
        new = (x_i-1)**2 * (1+10*np.sin(np.pi*x_i+1)**2)
        sum += new
    return term1+term3+sum

def ackley(x, a=20, b=0.2, c=2*np.pi):
    time.sleep(0.01)
    d = len(x)
    return -a*exp(-b*np.sqrt(sum(x**2)/d)) - exp(sum(np.cos(c*xi) for xi in x)/d) + a + exp(1)

def michalewicz(x, m=10):
    return -sum(np.sin(v)*np.sin(i*v^2/np.pi)^(2*m) for
               (i,v) in enumerate(x))

def rand_population_uniform(m,a,b):
    a = np.array(a)
    b = np.array(b)
    d = len(a)
    return [list(np.multiply(a+np.random.rand(d),(b-a))) for i in range(m)]

def TruncatedSelection(pop,tr):
    p = np.argsort(pop)
    return [p[np.random.randint(tr, size=(1,2)).ravel()] for i in p]

def SinglePointCrossover(a,b):
    n = len(a)
    i = int(np.random.randint(n))  
    print(i,a[:i],b[i:])#,np.concatenate(a[0:i],b[i:n+1]) )
    return a[:i]+b[i:n]

def GaussianMutation(child,sigma):
    if np.random.rand() < sigma:
        new_child = child + np.random.rand(len(child))*sigma
    else:
        new_child = child
    return new_child

def GeneticAlgorithm(f,population,k_max=10,sigma=0.1,trunc=10, parallel=False):
    n = len(population)
    for k in range(k_max):
        if parallel == True:
            pass
        else:
            f_pop = list([f(np.array(x)) for x in population])
        parents = np.array(TruncatedSelection(f_pop,trunc)).astype(int)
        
        children = [SinglePointCrossover(population[(p[0])],population[(p[1])]) for p in parents]
        
        population = [GaussianMutation(child,sigma) for child in children]
    
    return population[np.argsort(f(population))]
    


pop = rand_population_uniform(40, [0.0,0.0,0.0,0.0], [4.0,4.0,4.0,4.0])


F_pop = list([Levy3(np.array(x)) for x in pop])

parents = TruncatedSelection(F_pop, 16)
parents = np.array(parents).astype(int)

children = [SinglePointCrossover(pop[p[0]],pop[p[1]]) for p in parents]

population = [GaussianMutation(child,0.1) for child in children]

GA = GeneticAlgorithm(Levy3,pop)