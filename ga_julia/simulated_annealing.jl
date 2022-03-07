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

function simulated_annealing_p(f, x, t, k_max)
    y = f(x)
    x_best, y_best = x, y
    lk = ReentrantLock()
    
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
