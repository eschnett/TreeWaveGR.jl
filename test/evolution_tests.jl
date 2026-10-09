# The right-hand side, the integrator and the runs (`src/evolution.jl`,
# `src/stepping.jl`, `src/initialdata.jl`).

# The setup every test below starts from.
function wave_setup(T, case; N, roots, q=4, refined=false, coefficients=:auto)
    forest = wave_forest(T, case; N=N, roots=roots, refined=refined)
    fs = state_fieldset(forest, q)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(q)), case; q=q,
                    coefficients=coefficients)
    return forest, fs, p
end

"""
The right-hand side at the exact state at time `t`, against the exact time
derivative of the state there (a sixth-order central difference in `t`, of
the exact state, with step `1/64`): the truncation error of the spatial
operator alone, with no time integration and no accumulation.
"""
function rhs_error(T, case; N, roots, q=4, t=1 // 4, refined=false, coefficients=:auto)
    forest, fs, p = wave_setup(T, case; N=N, roots=roots, q=q, refined=refined,
                               coefficients=coefficients)
    tt = T(t)
    u = exact_statevector(fs, case, tt)
    du = similar(u)
    wave_rhs!(du, u, p, tt)
    δ = T(1 // 64)
    w = derivative_weights(T, Val(6), Val(1))
    dudt = zero(u)
    for (k, j) in enumerate(-3:3)
        j == 0 && continue
        dudt .+= w[k] .* exact_statevector(fs, case, tt + j * δ)
    end
    dudt ./= δ
    return (; err=volume_weighted_norm(fs, du .- dudt), h=minimum_spacing(T, forest),
            scale=volume_weighted_norm(fs, dudt))
end

# Cases with an exact solution on every kind of background, each on a domain
# that stays clear of every hole; the right-hand side's convergence on them
# is the test of the operator, coefficient by coefficient.
const RHS_CASES = [
    ("Minkowski, periodic, D=1",
     WaveCase(Background{1}(Minkowski()), PlaneWave(1.0, [2π, 0, 0]);
              extents=((0.0, 1.0),), periodic=(true,)), 1),
    ("superluminal constant shift, D=1",
     WaveCase(Background{1}(ConstantShift([1.5, 0, 0])), PlaneWave(1.0, [2π, 0, 0]);
              extents=((0.0, 1.0),)), 1),
    ("gauge wave, periodic, D=2",
     WaveCase(Background{2}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 2π, 0]);
              extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true)), 1),
    ("sine shift, periodic, D=2",
     WaveCase(Background{2}(SineShift(0.2, 1.0)), PlaneWave(1.0, [2π, 0, 0]);
              extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true)), 1),
    ("superluminal moving grid, D=2",
     WaveCase(Background{2}(MovingGrid(1.5, 0.3, 0.5)), PlaneWave(1.0, [2π, π, 0]);
              extents=((0.0, 1.0), (0.0, 1.0))), 1),
    ("skewed time, D=3",
     WaveCase(Background{3}(ShiftedMinkowski(0.4, 0.5)), PlaneWave(1.0, [π, π / 2, π / 2]);
              extents=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))), 1),
    ("river, D=2",
     WaveCase(Background{2}(River(1.0)), StaticRiver(1.0);
              extents=((2.5, 4.5), (-1.0, 1.0))), 1),
    ("Kerr–Schild a=0.6, D=3",
     WaveCase(Background{3}(KerrSchild(1.0, 0.6)), StaticHole(1.0, 0.6);
              extents=((2.5, 4.5), (2.5, 4.5), (-1.0, 1.0))), 1),
    ("boosted Kerr–Schild v=0.6, D=3",
     WaveCase(Background{3}(boost(KerrSchild(1.0, 0.0), [0.6, 0, 0])), StaticHole(1.0, 0.0);
              extents=((2.5, 4.5), (-1.0, 1.0), (0.5, 2.5))), 1),
]

@testset "The right-hand side converges at its order: $name, q=$q" for (name, case, roots) in
                                                                        RHS_CASES,
                                                                    q in (2, 4)
    D = ndims(case)
    Ns = D == 3 ? (8, 12, 16) : (16, 32, 64)
    rs = [rhs_error(Float64, case; N=N, roots=roots, q=q) for N in Ns]
    errs = [r.err for r in rs]
    rate = convergence_rate([r.h for r in rs], errs)
    @info "rhs" name q errs rate
    @test issorted(errs; rev=true)
    @test rate >= q - 1 // 4
end

@testset "The right-hand side keeps its order across a coarse-fine face: D=$D, q=$q" for D in
                                                                                       1:3,
                                                                                   q in (2, 4)
    # A wave number along a dropped axis would change the frequency without
    # changing the slice: at D = 1 the wave runs along x alone.
    k = [2π, D >= 2 ? 2π : 0, 0]
    case = WaveCase(Background{D}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, k);
                    extents=ntuple(_ -> (0.0, 1.0), D), periodic=ntuple(_ -> true, D))
    Ns = D == 3 ? (8, 12, 16) : (16, 32, 64)
    rs = [rhs_error(Float64, case; N=N, roots=2, q=q, refined=true) for N in Ns]
    rate = convergence_rate([r.h for r in rs], [r.err for r in rs])
    @info "interface rhs" D q rate
    @test rate >= q - 1 // 4
end

@testset "A polynomial solution's right-hand side is exact" begin
    # Every stencil here is exact on a quadratic, and the coefficients of a
    # constant shift are constant: the right-hand side is the exact time
    # derivative to rounding, on a two-level mesh, with dissipation on.
    for D in 1:3
        case = WaveCase(Background{D}(ConstantShift([0.5, D >= 2 ? 0.25 : 0.0, 0.0])),
                        PolynomialWave(1.0, 2.0); extents=ntuple(_ -> (0.0, 1.0), D),
                        ε=0.3)
        r = rhs_error(Float64, case; N=8, roots=2, refined=true)
        @test r.err <= 1e-12 * r.scale
    end
end

@testset "The sampled and the analytic coefficients give the same right-hand side: D=$D" for D in
                                                                                             1:3
    case = WaveCase(Background{D}(River(1.0)), StaticRiver(1.0);
                    extents=ntuple(d -> d == 1 ? (2.5, 4.5) : (-1.0, 1.0), D), ε=0.2)
    _, fs, ps = wave_setup(Float64, case; N=8, roots=2, refined=true, coefficients=:sampled)
    _, _, pa = wave_setup(Float64, case; N=8, roots=2, refined=true, coefficients=:analytic)
    @test TreeWaveGR.sampled(ps)
    @test !TreeWaveGR.sampled(pa)
    u = exact_statevector(fs, case, 0.25)
    dus, dua = similar(u), similar(u)
    wave_rhs!(dus, u, ps, 0.25)
    wave_rhs!(dua, u, pa, 0.25)
    # Agreement to rounding, not bit for bit: the sampled set was filled by
    # another kernel, which may fuse ForwardDiff's `muladd`s differently.
    @test dus ≈ dua rtol = 1e-13
    @test max_speed(ps, 0.25) ≈ max_speed(pa, 0.25) rtol = 1e-14
    # And `:auto` takes the sampled source for a stationary background only.
    @test TreeWaveGR.sampled(wave_setup(Float64, case; N=8, roots=2)[3])
    moving = WaveCase(Background{D}(boost(Minkowski(), [0.3, 0, 0])),
                      PlaneWave(1.0, [1.0, 0, 0]); extents=ntuple(_ -> (0.0, 1.0), D))
    @test !TreeWaveGR.sampled(wave_setup(Float64, moving; N=8, roots=1)[3])
    @test_throws ArgumentError wave_setup(Float64, moving; N=8, roots=1, coefficients=:sampled)
end

@testset "The right-hand side is linear on a periodic domain" begin
    case = WaveCase(Background{2}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 0, 0]);
                    extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true), ε=0.3)
    _, fs, p = wave_setup(Float64, case; N=8, roots=2, refined=true)
    rng = Random.MersenneTwister(1)
    u, v = randn(rng, statelength(fs)), randn(rng, statelength(fs))
    F(x) = wave_rhs!(similar(x), x, p, 0.125)
    @test F(2u .- 3v) ≈ 2F(u) .- 3F(v) rtol = 1e-12
    @test F(u) == F(u)                  # a pure function of (u, t)
    w = copy(u)
    F(u)
    @test u == w                        # `u` is not written
end

@testset "The maximum speed is the characteristic speed" begin
    case = WaveCase(Background{3}(Minkowski()), PlaneWave(1.0, [1.0, 0, 0]);
                    extents=ntuple(_ -> (0.0, 1.0), 3))
    @test max_speed(wave_setup(Float64, case; N=8, roots=1)[3], 0) == 1
    shifted = WaveCase(Background{2}(ConstantShift([1.5, -0.5, 0])),
                       PlaneWave(1.0, [1.0, 0, 0]); extents=((0.0, 1.0), (0.0, 1.0)))
    @test max_speed(wave_setup(Float64, shifted; N=8, roots=1)[3], 0) == 2.5
end

@testset "The right-hand side allocates nothing per point" begin
    case = WaveCase(Background{3}(boost(KerrSchild(1.0, 0.6), [0.3, 0, 0])),
                    StaticHole(1.0, 0.6); extents=((2.5, 4.5), (2.5, 4.5), (-1.0, 1.0)),
                    ε=0.1)
    allocs = map((8, 16)) do N
        _, fs, p = wave_setup(Float64, case; N=N, roots=1)
        u = exact_statevector(fs, case, 0)
        du = similar(u)
        launch_rhs!(du, p, 0.0)
        @allocated launch_rhs!(du, p, 0.0)
    end
    # The same handful of launch bookkeeping at 8³ and at 16³ points.
    @test allocs[1] == allocs[2]
end

@testset "A run converges at its order: $name, q=$q" for (name, case, roots, Ns, t_end) in [
        ("Minkowski, periodic, D=1",
         WaveCase(Background{1}(Minkowski()), PlaneWave(1.0, [2π, 0, 0]);
                  extents=((0.0, 1.0),), periodic=(true,)), 1, (16, 32, 64), 1 // 2),
        ("gauge wave, periodic, D=2",
         WaveCase(Background{2}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 2π, 0]);
                  extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true)), 1,
         (16, 32, 64), 1 // 2),
        ("Minkowski, periodic, D=3",
         WaveCase(Background{3}(Minkowski()), PlaneWave(1.0, [2π, 2π, 0]);
                  extents=ntuple(_ -> (0.0, 1.0), 3), periodic=ntuple(_ -> true, 3)), 1,
         (8, 12, 16), 1 // 4)],
    q in (2, 4)

    rs = [wave_errors(Float64, case; N=N, roots=roots, q=q, t_end=t_end) for N in Ns]
    errs = [r.l2 for r in rs]
    rate = convergence_rate([r.h for r in rs], errs)
    @info "run" name q errs rate
    @test issorted(errs; rev=true)
    @test rate >= q - 1 // 4
end

@testset "A slice that couples to a dropped axis is refused" begin
    @test_throws ArgumentError WaveCase(Background{1}(ConstantShift([0.5, 0.25, 0.0])),
                                        PolynomialWave(1.0, 2.0); extents=((0.0, 1.0),))
end

@testset "Dissipation damps grid-scale noise at a coarse-fine face" begin
    # Noise on a quiet state, on a periodic two-level mesh with a shift that
    # is superluminal along x. The first unit of time is a transient — noise
    # in `u` feeds `Π` at amplitude `1/h` — and is not measured. Over the next
    # two, without dissipation the noise grows at the coarse-fine face (×2.2
    # measured; a uniform mesh, or a subluminal shift, does not grow), and
    # with `ε = 0.5` it decays (×0.61) (`CODE.md`, "Measured results").
    case = WaveCase(Background{2}(ConstantShift([1.5, 0.5, 0])), PolynomialWave(0.0, 1.0);
                    extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true))
    ratios = map((0.0, 0.5)) do ε
        _, fs, p = wave_setup(Float64, with_dissipation(case, ε); N=16, roots=2,
                              refined=true)
        u = 1e-8 .* randn(Random.MersenneTwister(2), statelength(fs))
        u = wave_solve(p, u, 0, 1, wave_steps(p, 0, 1))
        n1 = volume_weighted_norm(fs, u)
        u = wave_solve(p, u, 1, 3, wave_steps(p, 1, 3))
        volume_weighted_norm(fs, u) / n1
    end
    @info "noise growth over t = 1 … 3" ratios
    @test ratios[1] > 1.5
    @test ratios[2] < 0.75
end

@testset "The convergence rate takes any floating-point type" begin
    hs = Float32x2[1 // 4, 1 // 8, 1 // 16]
    @test convergence_rate(hs, hs .^ 4) ≈ 4
    @test convergence_rate(Float32[0.5, 0.25], Float32[1, 1 // 4]) ≈ 2
end

@testset "A case is checked when it is built" begin
    # A boost along a dropped axis leaves Minkowski's metric alone (to
    # rounding) but mixes `z` into the solution: refused by the wave-equation
    # probe at every speed, not by accident of rounding at some.
    for v in (0.5, 0.6, 0.8)
        @test_throws ArgumentError WaveCase(Background{2}(boost(Minkowski(), [0, 0, v])),
                                            PlaneWave(1.0, [1.0, 0, 0]);
                                            extents=((0.0, 1.0), (0.0, 1.0)))
    end
    # A boost in the plane is a reduction, and is accepted.
    @test WaveCase(Background{2}(boost(Minkowski(), [0.5, 0.3, 0])),
                   PlaneWave(1.0, [1.0, 0, 0]); extents=((0.0, 1.0), (0.0, 1.0))) isa
          WaveCase
    @test WaveCase(Background{1}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 0, 0]);
                   extents=((0.0, 1.0),), periodic=(true,)) isa WaveCase
    # A wave number along a dropped axis changes only the frequency.
    @test_throws ArgumentError WaveCase(Background{1}(Minkowski()),
                                        PlaneWave(1.0, [2π, 1.0, 0]); extents=((0.0, 1.0),))
    # Negative dissipation is an amplifier.
    @test_throws ArgumentError WaveCase(Background{1}(Minkowski()),
                                        PlaneWave(1.0, [2π, 0, 0]); extents=((0.0, 1.0),),
                                        ε=-0.1)
end

@testset "A rational parameter is retyped" begin
    case = WaveCase(Background{3}(KerrSchild(1 // 1, 3 // 5)), StaticHole(1.0, 0.6);
                    extents=((3.0, 5.0), (-1.0, 1.0), (-1.0, 1.0)))
    c32 = retype(Float32, case)
    @test c32.background.metric isa KerrSchild{Float32}
    @test retype(Float64, case).background.metric isa KerrSchild{Float64}
end
