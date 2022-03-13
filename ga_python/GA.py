"""This module implements Genetic Algorithms."""
import numpy as np
from multiprocessing import Pool


def rand_population_uniform(m, a, b):
    """
    Creates a starting population of size `m` using a uniform random
    distribution, where `a` is the lower bound, and `b` is the upper bound
    for the design variables.
    """
    a = np.array(a)
    b = np.array(b)
    d = len(a)
    return [list(np.multiply(a + np.random.rand(d), (b-a))) for i in range(m)]


def TruncatedSelection(pop, tr):
    """Performs truncation selection for top `tr` candidates."""
    p = np.argsort(pop)
    return [p[np.random.randint(tr, size=(1, 2)).ravel()] for i in p]


def SinglePointCrossover(a, b):
    """Performs a single point crossover for two parents, `a` and `b`."""
    a = list(a)
    b = list(b)
    n = len(a)
    i = np.random.randint(n+1)
    return a[:i]+b[i:]


def GaussianMutation(child, sigma):
    """
    Performs a Gaussian mutation for a child candidate, with scaling factor
    `sigma`.
    """
    if np.random.rand() < sigma:
        new_child = child + np.random.rand(len(child))*sigma
    else:
        new_child = child
    return new_child


def GeneticAlgorithm(f, population, k_max=10, sigma=0.1, trunc=10,
                     processes=None):
    """
    Performs a Genetic Algorithm optimization with optional parallelization.

    Args:
        f (Function): Objective function.
        population (list of lists): A list of design variable combinations
        representing an initial population.
        k_max (int): Maximum iterations.
        sigma (float): Scaling factor for Gaussian mutation.
        trunc (int): Cutoff for truncation selection.
        processes (int): Number of processes in the `Pool`.

    Returns:
        Array: An Array representing the minimizers for the given objective
        function.
    """
    if processes:
        p = Pool(processes=processes)

        for _ in range(k_max):
            results = []
            for x in population:
                res = p.apply_async(f, [x])
                results.append(res)

            f_pop = [r.get() for r in results]

            parents = np.array(TruncatedSelection(f_pop, trunc))
            children = [
                SinglePointCrossover(population[p[0]],
                                     population[p[1]]) for p in parents]

            population = [GaussianMutation(child, sigma) for child in children]

    else:
        for k in range(k_max):
            f_pop = list([f(np.array(x)) for x in population])
            parents = np.array(TruncatedSelection(f_pop, trunc))

            children = [
                SinglePointCrossover(population[p[0]],
                                     population[p[1]]) for p in parents]

            population = [GaussianMutation(child, sigma) for child in children]

    f_pop = list([f(np.array(x)) for x in population])
    return population[np.argmin(f_pop)]
