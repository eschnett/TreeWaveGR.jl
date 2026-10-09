# Thread-count independence: every assertion is exact equality. Roundoff-level
# agreement is what a reassociated sum gives, and a reassociated sum is the bug.

include("thread_workload.jl")

@testset "A run is bit-identical across thread counts" begin
    reference = thread_digests()
    @test length(reference) == 6
    other = Threads.nthreads() == 1 ? max(2, min(4, Sys.CPU_THREADS)) : 1
    script = joinpath(@__DIR__, "thread_workload.jl")
    # The active project, not `test/`: under `Pkg.test` the tests run in a
    # sandbox and `test/Project.toml` has no manifest of its own.
    project = Base.active_project()
    out = read(`$(Base.julia_cmd()) --threads=$other --project=$project $script`, String)
    @test split(chomp(out), '\n') == reference
end
