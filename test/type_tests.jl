# The run in the caller's floating-point type. `Float32` catches a `Float64`
# leaking into per-point arithmetic; `Float32x2` (MultiFloats) catches a
# dependence on a hardware float. MultiFloats has no transcendental
# functions, so the software type runs the algebraic case: a constant shift
# (the coefficients need only `sqrt`) and the polynomial wave. Its error
# against the exact solution is not rounding — RK4's stage order is one, and
# the Dirichlet data depend on time — but it is the same run at every
# precision, so each type must reproduce a `BigFloat` run of it to its own
# precision. (The reference is rounded to `Float64` for the comparison, so
# `Float32x2` is held to `1000 eps(Float32x2)`, about `1e-11`: far better than
# `Float32` could do, and not quite at `Float32x2`'s own limit.)

function polynomial_run(T)
    case = WaveCase(Background{2}(ConstantShift([0.5, 0.25, 0.0])),
                    PolynomialWave(1.0, 2.0); extents=((0.0, 1.0), (0.0, 1.0)), ε=0.25)
    forest = wave_forest(T, case; N=8, roots=2, refined=true)
    fs = state_fieldset(forest, 4)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(4)), case; q=4)
    u0 = exact_statevector(fs, case, 0)
    u1 = wave_solve(p, u0, 0, 1 // 4, 16)
    return fs, p, u0, u1
end

const POLYNOMIAL_REFERENCE = let
    fs, _, _, u1 = polynomial_run(BigFloat)
    Float64.(u1)
end

@testset "A run in $T is a run in $T" for T in (Float32, Float64, Float32x2)
    fs, p, u0, u1 = polynomial_run(T)
    @test eltype(fs.work) === T
    @test TreeWaveGR.sampled(p)
    @test eltype(p.cfs.work) === T
    @test eltype(u0) === T
    @test eltype(u1) === T
    @test volume_weighted_norm(fs, u1) isa T
    diff = maximum(abs, TreeWaveGR.tofloat64.(u1) .- POLYNOMIAL_REFERENCE)
    scale = maximum(abs, POLYNOMIAL_REFERENCE)
    @info "precision" T diff / scale
    @test diff <= 1000 * TreeWaveGR.tofloat64(eps(T)) * scale
end

@testset "A curved background runs in $T" for T in (Float32, Float64)
    case = WaveCase(Background{3}(boost(KerrSchild(1.0, 0.6), [0.3, 0, 0])),
                    StaticHole(1.0, 0.6); extents=((2.5, 4.5), (2.5, 4.5), (-1.0, 1.0)))
    forest = wave_forest(T, case; N=8, roots=1)
    fs = state_fieldset(forest, 4)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(4)), case; q=4)
    u1 = wave_solve(p, exact_statevector(fs, case, 0), 0, 1 // 8, 4)
    @test eltype(u1) === T
    err = volume_weighted_norm(fs, u1 .- exact_statevector(fs, case, 1 // 8))
    @test err < 1e-4
end
