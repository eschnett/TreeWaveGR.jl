using Test
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using MultiFloats
using Random
using SpacetimeMetrics
using StaticArrays
using TreeAMR
using TreeWaveGR

# The suite is expected to pass, with identical numbers, at any thread count.
# `Pkg.test` does not inherit `-t`, so the thread count has to be passed
# explicitly — see `CLAUDE.md`.
@info "Running the tests on $(Threads.nthreads()) thread(s)"

@testset verbose = true "TreeWaveGR.jl" begin
    include("stencils_tests.jl")
    include("backgrounds_tests.jl")
    include("exact_tests.jl")
    include("evolution_tests.jl")
    include("hole_tests.jl")
    include("driver_tests.jl")
    include("type_tests.jl")
    include("threading_tests.jl")
end
