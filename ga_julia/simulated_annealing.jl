"""
Performs a simple single-threaded Simulated Annealing optimization.

  Args:
    f (Function): Objective function.
    x (Array): Initial design variable values.
    t (Function): A function representing the annealing schedule (temperature).
    k_max (Integer): Maximum iterations.

  Returns:
    Tuple(Array, Float): A tuple representing the minimizers and minimum for
      the given objective function.
"""
function simulated_annealing(f, x, t, k_max)
    y = f(x)
    x_best, y_best = x, y
    for k in 1 : k_max
        x′ = x + randn(length(x))
        y′ = f(x′)
        Δy = y′ - y
        if Δy ≤ 0 || rand() < exp(-Δy/t(k))
            x, y = x′, y′
        end
        if y′ < y_best
            x_best, y_best = x′, y′
        end
    end
    return x_best, y_best
end


"""
Performs a simple multithreaded Simulated Annealing optimization.

  Args:
    f (Function): Objective function.
    x (Array): Initial design variable values.
    t (Function): A function representing the annealing schedule (temperature).
    k_max (Integer): Maximum iterations.

  Returns:
    Tuple(Array, Float): A tuple representing the minimizers and minimum for
      the given objective function.
"""
function simulated_annealing_p(f, x, t, k_max)
    y = f(x)
    x_best, y_best = x, y
    # create a lock to use in reading/writing current best solution
    lk = ReentrantLock()
    
    # split max iterations evenly over threads
    Threads.@threads for k in 1 : k_max
        x′ = x + randn(length(x))
        y′ = f(x′)
        Δy = y′ - y
        if Δy ≤ 0 || rand() < exp(-Δy/t(k))
            x, y = x′, y′
        end
        
        # acquire lock to potentially update current best while avoiding race conditions
        lock(lk) do
            if y′ < y_best
                x_best, y_best = x′, y′
            end
        end
    end
    return x_best, y_best
end
