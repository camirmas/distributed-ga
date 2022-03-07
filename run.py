import numpy as np
import time

from ga_python.simulated_annealing import simulated_annealing, sa_parallel
from ga_python.functions import ackley

np.random.seed(0)

def run():
    pass

def run_benchmark(fn, args):
    start = time.perf_counter()
    res = fn(*args)
    print(f"Completed Execution in {time.perf_counter() - start} seconds")
    print(res)
    return res

if __name__ == '__main__':
    x = np.array([3.0, 3.0])
    k_max = 1000

    print("\nBenchmarking Simulated Annealing baseline (Python)...")
    run_benchmark(simulated_annealing, (ackley, x.copy(), k_max))

    state = {}
    print("\nBenchmarking Simulated Annealing parallelized (Python)...")
    run_benchmark(sa_parallel, (ackley, x.copy(), k_max, state))
