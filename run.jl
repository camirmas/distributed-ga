using Random, BenchmarkTools

include("ga_julia/simulated_annealing.jl")
include("ga_julia/functions.jl")

Random.seed!(0)

x = [3.0, 3.0]
k_max = 1000

# fast annealing
T0 = 25
t(k) = T0/k

println("\nBenchmarking Simulated Annealing baseline (Julia)...")
@btime simulated_annealing($ackley, $(copy(x)), $t, $k_max) samples=3

println("\nBenchmarking Simulated Annealing parallelized (Julia)...")
@btime simulated_annealing_p($ackley, $(copy(x)), $t, $k_max) samples=3
