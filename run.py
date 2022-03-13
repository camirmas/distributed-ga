import numpy as np
import time

from ga_python.simulated_annealing import simulated_annealing, sa_parallel
from ga_python.functions import ackley, Levy3
from ga_python.GA import rand_population_uniform, GeneticAlgorithm

np.random.seed(0)


def _run_benchmark(fn, args, kwargs=None):
    start = time.perf_counter()
    kwargs = kwargs or {}
    res = fn(*args, **kwargs)
    print(f"Completed Execution in {time.perf_counter() - start} seconds")
    return res


if __name__ == '__main__':
    # SA
    x = np.array([3.0, 3.0])
    k_max = 1000

    print("\nBenchmarking Simulated Annealing baseline (Python)...")
    _run_benchmark(simulated_annealing, (ackley, x.copy(), k_max))

    for i in [5, 10, 20]:
        print(f'\nBenchmarking Simulated Annealing (Python, {i} processes)...')
        _run_benchmark(sa_parallel, (ackley, x.copy(), k_max, i))

    # GA

    pop = rand_population_uniform(40, [0.0, 0.0], [4.0, 4.0])

    print("\nBenchmarking Genetic Algorithm baseline (Python)...")
    _run_benchmark(GeneticAlgorithm, (Levy3, pop.copy()))

    for i in [5, 10, 20]:
        print(f'\nBenchmarking Genetic Algorithm (Python, {i} processes)...')
        _run_benchmark(GeneticAlgorithm, (Levy3, pop.copy()), {'processes': i})
