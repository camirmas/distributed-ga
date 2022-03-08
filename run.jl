using Random

include("ga_julia/simulated_annealing.jl")
include("ga_julia/functions.jl")
include("ga_julia/GA.jl")

Random.seed!(0)

x = [3.0, 3.0]
k_max = 1000

# fast annealing
T0 = 25
t(k) = T0/k

threads = Threads.nthreads()

if threads == 1
    println("\nBenchmarking Simulated Annealing baseline (Julia)...")
    @time simulated_annealing(ackley, copy(x), t, k_max)
else
    println("\nBenchmarking Simulated Annealing (Julia, $(threads) threads)...")
    @time simulated_annealing_p(ackley, copy(x), t, k_max)
end

# GA

f = x -> levy3(x)

S = TruncationSelection(10)
C = SinglePointCrossover()
M = GaussianMutation(0.1)

k_max = 10
m = 40
population = rand_population_uniform(m, [0.0, 0.0], [4.0, 4.0])

if threads == 1
    println("\nBenchmarking Genetic Algorithm baseline (Julia)...")
    @time genetic_algorithm(f, population, k_max, S, C, M)
else
    println("\nBenchmarking Genetic Algorithm (Julia, $(threads) threads)...")
    @time genetic_algorithm(f, population, k_max, S, C, M; parallel=true)
end
