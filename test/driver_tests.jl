# The refinement criterion, the driver and checkpoints (`src/refinement.jl`,
# `src/driver.jl`, `src/checkpoint.jl`).

@testset "The Löhner indicator" begin
    @test lohner(1.0, 2.0, 3.0, 1.0) == 0               # linear data
    @test lohner(0.0, 1.0, 0.0, 1.0) ≈ 2 / (2 + 0.04)   # a spike
    @test 0 <= lohner(-1.0, 1.0, -1.0, 0.0) <= 1
    @test lohner(0.0, 0.0, 0.0, 0.0) == 0               # no NaN
    # The floor is global: dust far below the amplitude scores nothing.
    @test lohner(1e-16, 0.0, 1e-16, 1.0) < 1e-12
    @test lohner(1.0f0, 2.0f0, 3.0f0, 1.0f0) isa Float32
end

@testset "The criterion's parameters are checked" begin
    @test_throws ArgumentError Refinement(Float64, Val(2); refine_tol=0.1, coarsen_tol=0.1,
                                          maxlevel_cap=2)
    @test_throws ArgumentError Refinement(Float64, Val(2); refine_tol=0.1, coarsen_tol=0.01,
                                          maxlevel_cap=1, floor_level=2)
    forest = Forest{Float64}((2, 2); N=8)
    @test refinement_buffer(forest, 1, 0.1) == ceil(Int, 0.1 / spacing(forest, 1)) + 1
    @test_throws ArgumentError refinement_buffer(forest, 1, 1.0)
end

@testset "The floor holds the blocks in its ball, and moves" begin
    forest = Forest{Float64}((4, 4); N=8, extents=((0.0, 4.0), (0.0, 4.0)))
    crit = Refinement(Float64, Val(2); refine_tol=1, coarsen_tol=0.5, maxlevel_cap=2,
                      floor_level=2, floor_center=(0.5, 0.5), floor_radius=0.25,
                      floor_velocity=(1.0, 0.0))
    levels(t) = [TreeWaveGR.floor_level(crit, forest, k, t) for k in forest.leaves]
    @test count(==(2), levels(0)) == 1                  # the block around (0.5, 0.5)
    @test count(==(2), levels(0.5)) == 2                # the ball spans x = 1
    @test count(==(2), levels(2)) == 1
    @test levels(2) != levels(0)
end

# A pulse crossing the unit square diagonally, and the criterion that tracks
# it two levels deep.
const PULSE_CASE = WaveCase(Background{2}(Minkowski()),
                            PlanePulse(1.0, [1.0, 0.5, 0], 0.08, 0.3);
                            extents=((0.0, 1.0), (0.0, 1.0)), ε=0.25)
const PULSE_CRITERION = Refinement(Float64, Val(2); refine_tol=0.05, coarsen_tol=0.01,
                                   maxlevel_cap=2)

@testset "An adaptive run is the uniform run at the pulse, on fewer points" begin
    t_end = 1 // 4
    adaptive = evolve!(Float64, PULSE_CASE; N=8, roots=4, t_end=t_end, chunk=1 // 32,
                       refinement=PULSE_CRITERION)
    uniform = evolve!(Float64, PULSE_CASE; N=32, roots=4, t_end=t_end)
    @test length(adaptive.records) == 8
    @test adaptive.passes >= 2
    @test adaptive.nregrids >= 1
    @test all(r -> r.maxlevel == 2, adaptive.records)
    ea, eu = adaptive.records[end].l2, uniform.records[end].l2
    @info "pulse" ea eu nblocks = [r.nblocks for r in adaptive.records]
    @test ea <= 1.05 * eu
    # 4² roots refined twice would be 256 blocks.
    @test maximum(r -> r.nblocks, adaptive.records) < 256
    @test adaptive.t == t_end
end

@testset "A fixed-mesh run is a sequence of solves" begin
    case = WaveCase(Background{2}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 2π, 0]);
                    extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true))
    r = evolve!(Float64, case; N=16, roots=1, t_end=1 // 4, chunk=1 // 8)
    @test length(r.records) == 2
    @test r.nregrids == 0
    # The same steps as one `wave_errors` run, in two pieces: the same error
    # to rounding (each chunk sizes its own steps).
    w = wave_errors(Float64, case; N=16, roots=1, t_end=1 // 4)
    @test r.records[end].l2 ≈ w.l2 rtol = 0.05
    seen = Float64[]
    evolve!(Float64, case; N=16, roots=1, t_end=1 // 4, chunk=1 // 8,
            observer=(p, t, u, row) -> push!(seen, row.t))
    @test seen == [0.125, 0.25]
end

@testset "A moving floor refines ahead of itself" begin
    # A boosted hole's case, with a floor ball that moves along x at 4.8: the
    # regrid after the first chunk (t = 1/4) refines where the ball has moved
    # to, about x = 5.2.
    hole = WaveCase(Background{3}(boost(KerrSchild(1.0, 0.0), [0.6, 0, 0])),
                    StaticHole(1.0, 0.0); extents=((3.0, 7.0), (-2.0, 2.0), (-2.0, 2.0)),
                    ε=0.25)
    crit = Refinement(Float64, Val(3); refine_tol=1, coarsen_tol=0.5, maxlevel_cap=1,
                      floor_level=1, floor_center=(4.0, 0.0, 0.0), floor_radius=0.25,
                      floor_velocity=(4.8, 0.0, 0.0))
    refined_x(forest) = sort(unique([block_origin(forest, k)[1] for k in forest.leaves
                                     if level(k) == 1]))
    first = Ref{Vector{Float64}}()
    r = evolve!(Float64, hole; N=8, roots=(2, 2, 2), t_end=1 // 2, chunk=1 // 4,
                refinement=crit, buffer=0,
                observer=(p, t, u, row) -> row.chunk == 1 && (first[] = refined_x(p.fs.forest)))
    @test r.records[1].maxlevel == 1
    @test first[] == [3.0, 4.0]          # the ball about x = 4 at t = 0
    @test r.nregrids == 1
    @test 5.0 in refined_x(r.forest)      # about x = 5.2 after the first chunk
    @test r.records[end].l2 < 1e-3
end

@testset "A restart continues the run bit for bit" begin
    mktempdir() do dir
        prefix = joinpath(dir, "pulse")
        kw = (; N=8, roots=4, t_end=1 // 8, chunk=1 // 32, refinement=PULSE_CRITERION)
        full = evolve!(Float64, PULSE_CASE; kw..., checkpoint=prefix)
        @test all(c -> isfile(checkpoint_path(prefix, c)), 1:4)
        run = load_run(checkpoint_path(prefix, 2))
        @test run.data.chunk == 2
        rest = evolve!(Float64, PULSE_CASE; kw..., restart=checkpoint_path(prefix, 2))
        @test length(rest.records) == 2
        @test rest.t == full.t
        @test rest.forest.leaves == full.forest.leaves
        @test rest.u == full.u
        @test rest.nsteps == full.nsteps
        @test [r.l2 for r in rest.records] == [r.l2 for r in full.records[3:4]]
        # A restart at another order is refused.
        @test_throws ArgumentError evolve!(Float64, PULSE_CASE; kw..., q=2,
                                           restart=checkpoint_path(prefix, 2))
    end
end

@testset "The chunk count is not fooled by rounding" begin
    # `0.07/0.01` is `7.000000000000001`, and `0.33/0.03` is a few ulps above
    # 11: a bare `ceil` gave an empty eighth chunk, and a twelfth a few ulps
    # long.
    @test TreeWaveGR.chunk_count(0.07, 0.01) == 7
    @test TreeWaveGR.chunk_count(0.33, 0.03) == 11
    @test TreeWaveGR.chunk_count(0.5, 0.2) == 3
    @test TreeWaveGR.chunk_count(1.0, 1.0) == 1
    @test TreeWaveGR.chunk_count(0.25, 1.0) == 1
    for T in (Float32, Float64), a in 1:100, b in 1:48
        t_end, chunk = T(b * a // 100), T(a // 100)
        n = TreeWaveGR.chunk_count(t_end, chunk)
        ends = [TreeWaveGR.chunk_end(c, chunk, t_end) for c in 1:n]
        @test issorted(ends) && allunique(ends)
        @test ends[end] == t_end
        @test n == b
    end
    case = WaveCase(Background{1}(Minkowski()), PlaneWave(1.0, [2π, 0, 0]);
                    extents=((0.0, 1.0),), periodic=(true,))
    r = evolve!(Float64, case; N=16, roots=1, t_end=0.07, chunk=0.01)
    @test length(r.records) == 7
    @test r.t == 0.07
    @test_throws ArgumentError evolve!(Float64, case; N=16, roots=1, t_end=0)
end

@testset "The observer and the caller see the evolved state" begin
    seen = Ref(0.0)
    r = evolve!(Float64, PULSE_CASE; N=8, roots=4, t_end=1 // 16, chunk=1 // 32,
                refinement=PULSE_CRITERION,
                observer=(p, t, u, row) -> begin
                    v = similar(u)
                    gather!(v, p.fs)
                    seen[] = max(seen[], maximum(abs, v .- u))
                end)
    @test seen[] == 0
    v = similar(r.u)
    gather!(v, r.fs)
    @test v == r.u
end

@testset "A criterion in another type drives the run" begin
    crit32 = Refinement(Float32, Val(2); refine_tol=0.05, coarsen_tol=0.01, maxlevel_cap=2)
    r = evolve!(Float64, PULSE_CASE; N=8, roots=4, t_end=1 // 32, chunk=1 // 32,
                refinement=crit32)
    @test r.records[end].maxlevel == 2
    r32 = evolve!(Float32, PULSE_CASE; N=8, roots=4, t_end=1 // 32, chunk=1 // 32,
                  refinement=PULSE_CRITERION)
    @test eltype(r32.u) === Float32
    @test r32.forest.leaves == r.forest.leaves
end

@testset "A restart keeps the adaptation's pass count" begin
    mktempdir() do dir
        prefix = joinpath(dir, "pulse")
        kw = (; N=8, roots=4, t_end=1 // 16, chunk=1 // 32, refinement=PULSE_CRITERION)
        full = evolve!(Float64, PULSE_CASE; kw..., checkpoint=prefix)
        rest = evolve!(Float64, PULSE_CASE; kw..., restart=checkpoint_path(prefix, 1))
        @test full.passes >= 2
        @test rest.passes == full.passes
    end
end
