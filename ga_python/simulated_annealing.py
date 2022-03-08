from numpy.random import rand, randn
from math import exp
from multiprocessing import Pool, Manager


# fast annealing
T0 = 25


def t(k):
    return T0/(k+1)


def simulated_annealing(f, x, k_max):
    y = f(x)
    x_best, y_best = x, y

    for k in range(k_max):
        x_new = x + randn(len(x))
        y_new = f(x_new)
        Δy = y_new - y
        if Δy <= 0 or rand() < exp(-Δy/t(k)):
            x, y = x_new, y_new

        if y_new < y_best:
            x_best, y_best = x_new, y_new

    return x_best, y_best


def run_sa(f, t, k, state):
    x, y = state['x'], state['y']
    y_best = state['y_best']

    x_new = x + randn(len(x))
    y_new = f(x_new)
    Δy = y_new - y
    if Δy <= 0 or rand() < exp(-Δy/t(k)):
        state['x'], state['y'] = x_new, y_new

    if y_new < y_best:
        state['x_best'], state['y_best'] = x_new, y_new


def sa_parallel(f, x, k_max):
    with Manager() as manager:
        state = manager.dict()
        y = f(x)
        state['x'] = x
        state['x_best'] = x
        state['y'] = y
        state['y_best'] = y

        p = Pool(processes=10)
        results = []
        for k in range(k_max):
            r = p.apply_async(run_sa, (f, t, k, state))
            results.append(r)

        [r.get() for r in results]

        return state['x_best'], state['y_best']
