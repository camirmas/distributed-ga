# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:29:54 2022

@author: mankleh
"""

import numpy as np
from multiprocessing import Pool, Manager


def rand_population_uniform(m,a,b):
    a = np.array(a)
    b = np.array(b)
    d = len(a)
    return [list(np.multiply(a+np.random.rand(d),(b-a))) for i in range(m)]


def TruncatedSelection(pop, tr):
    p = np.argsort(pop)
    #print([p[np.random.randint(tr, size=(1,2)).ravel()] for i in p])
    return [p[np.random.randint(tr, size=(1,2)).ravel()] for i in p]


def SinglePointCrossover(a,b):
    a = list(a)
    b = list(b)
    n = len(a)
    i = np.random.randint(n+1)
    # print(i,a[:i],b[i:],a[:i]+b[i:])#np.concatenate(a[:i],b[i:n]) )
    return a[:i]+b[i:]


def GaussianMutation(child,sigma):
    if np.random.rand() < sigma:
        new_child = child + np.random.rand(len(child))*sigma
    else:
        new_child = child
    return new_child


def GeneticAlgorithm(f, population, k_max=10, sigma=0.1, trunc=10, parallel=False, threads=10):
    if parallel:
        p = Pool(processes=threads)

        for k in range(k_max):
            results = []
            for x in population:
                res = p.apply_async(f, [x])
                results.append(res)

            f_pop = [r.get() for r in results]

            parents = np.array(TruncatedSelection(f_pop, trunc))
            children = [SinglePointCrossover(population[p[0]], population[p[1]]) for p in parents]

            population = [GaussianMutation(child, sigma) for child in children]

    else:
        for k in range(k_max):
            f_pop = list([f(np.array(x)) for x in population])
            parents = np.array(TruncatedSelection(f_pop, trunc))

            children = [SinglePointCrossover(population[p[0]], population[p[1]]) for p in parents]

            population = [GaussianMutation(child, sigma) for child in children]

    f_pop = list([f(np.array(x)) for x in population])
    return population[np.argmin(f_pop)]
