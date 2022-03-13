"""
Creates a starting population of size `m` using a uniform random distribution,
where `a` is the lower bound, and `b` is the upper bound for the design
variables.
"""
function rand_population_uniform(m, a, b)
    d = length(a)
    return [a+rand(d).*(b-a) for i in 1:m]
end

abstract type SelectionMethod end
struct TruncationSelection <: SelectionMethod
    k # top k to keep
end
function select(t::TruncationSelection, y)
    p = sortperm(y)
    return [p[rand(1:t.k, 2)] for i in y]
end


abstract type CrossoverMethod end
struct SinglePointCrossover <: CrossoverMethod end
function crossover(::SinglePointCrossover, a, b)
    i = rand(1:length(a))
    return vcat(a[1:i], b[i+1:end])
end

abstract type MutationMethod end
struct GaussianMutation <: MutationMethod
    σ
end
function mutate(M::GaussianMutation, child)
    return child + randn(length(child))*M.σ
end


"""
Performs a Genetic Algorithm optimization with optional parallelization.

  Args:
    f (Function): Objective function.
    population (Array[]): An Array of design variable combinations
      representing an initial population.
    k_max (Integer): Maximum iterations
    S (SelectionMethod): The selection method to be used.
    C (CrossoverMethod): The crossover method to be used.
    M (MutationMethod): The mutation method to be used.
    parallel (Boolean, optional): Optionally use multithreading.

  Returns:
    Array: An Array representing the minimizers for the given objective
      function.
"""
function genetic_algorithm(f, population, k_max, S, C, M; parallel=false)
    n = length(population)
    for k in 1 : k_max
        if parallel
            # Perform function evaluations in parallel
            f_pop = zeros(n)
            Threads.@threads for i=1:n
               f_pop[i] = f(population[i]) 
            end
        else
            f_pop = f.(population)
        end
        # proceed as normal for selection, crossover, mutation
        parents = select(S, f_pop)
        children = [crossover(C, population[p[1]], population[p[2]]) for p in parents]
        population .= mutate.(Ref(M), children)
    end

    if parallel
        # Perform function evaluations in parallel
        f_pop = zeros(n)
        Threads.@threads for i=1:n
            f_pop[i] = f(population[i]) 
        end
    else
        f_pop = f.(population)
    end
    population[argmin(f_pop)]
end
