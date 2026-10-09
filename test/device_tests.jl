# The run on a device (`CODE.md`, "Precision, threads, devices").
#
# Opt-in: `TREEWAVEGR_TEST_BACKEND=metal` or `=cuda` runs these against that
# device, from an environment that has the device package (`CLAUDE.md`,
# "Commands"); unset, they run on `CPU()`, which is what CI does — the same
# code paths, compared against themselves.

const DEVICE = get(ENV, "TREEWAVEGR_TEST_BACKEND", "cpu")
if DEVICE == "metal"
    using Metal
elseif DEVICE == "cuda"
    using CUDA
end
const DEVICE_BACKEND = DEVICE == "metal" ? Metal.MetalBackend() :
                       DEVICE == "cuda" ? CUDA.CUDABackend() : CPU()
# Metal has no hardware `Float64`.
const DEVICE_T = DEVICE == "metal" ? Float32 : Float64
@info "Device tests on $DEVICE_BACKEND in $DEVICE_T"

const DEVICE_CASES = [
    ("gauge wave, analytic",
     WaveCase(Background{2}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 2π, 0]);
              extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true), ε=0.25), :analytic),
    ("hole, sampled",
     WaveCase(Background{3}(KerrSchild(1.0, 0.6)), StaticHole(1.0, 0.6);
              extents=((3.0, 5.0), (-1.0, 1.0), (-1.0, 1.0)), ε=0.25), :sampled),
    ("boosted hole, analytic",
     WaveCase(Background{3}(boost(KerrSchild(1.0, 0.6), [0.6, 0, 0])), StaticHole(1.0, 0.6);
              extents=((3.0, 5.0), (-1.0, 1.0), (-1.0, 1.0)), ε=0.25), :analytic),
]

# The right-hand side at the exact state and a short solve, on `backend`,
# brought back to the host.
function device_run(backend, case, coefficients)
    T = DEVICE_T
    forest = wave_forest(T, case; N=8, roots=2, refined=true)
    fs = state_fieldset(forest, 4; backend=backend)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(4)), case; q=4,
                    coefficients=coefficients)
    @test get_backend(p.spacings) == backend
    u = exact_statevector(fs, case, 0)
    du = similar(u)
    wave_rhs!(du, u, p, 0)
    u1 = wave_solve(p, u, 0, 1 // 8, 4)
    return (; du=Array(du), u=Array(u), u1=Array(u1), h=minimum_spacing(T, forest),
            speed=max_speed(p, 0))
end

@testset "The device runs the host's run: $name" for (name, case, coefficients) in DEVICE_CASES
    host = device_run(CPU(), case, coefficients)
    dev = device_run(DEVICE_BACKEND, case, coefficients)
    T = DEVICE_T
    @test dev.u ≈ host.u rtol = 100 * eps(T)
    # The right-hand side of a static solution is cancellation alone, so it
    # is compared against the size of its terms, `|u|/h²`.
    scale = maximum(abs, host.u) / host.h^2
    @test maximum(abs, dev.du .- host.du) <= 1000 * eps(T) * scale
    @test maximum(abs, dev.u1 .- host.u1) <= 1000 * eps(T) * maximum(abs, host.u1)
    @test dev.speed ≈ host.speed rtol = 100 * eps(T)
end

@testset "An adaptive run on the device chooses the host's mesh" begin
    T = DEVICE_T
    case = WaveCase(Background{2}(Minkowski()), PlanePulse(1.0, [1.0, 0.5, 0], 0.08, 0.3);
                    extents=((0.0, 1.0), (0.0, 1.0)), ε=0.25)
    crit = Refinement(T, Val(2); refine_tol=0.05, coarsen_tol=0.01, maxlevel_cap=2)
    kw = (; N=8, roots=4, t_end=1 // 16, chunk=1 // 32, refinement=crit)
    host = evolve!(T, case; kw...)
    dev = evolve!(T, case; kw..., backend=DEVICE_BACKEND)
    @test dev.forest.leaves == host.forest.leaves
    @test dev.passes == host.passes
    @test dev.nregrids == host.nregrids
    @test Array(dev.u) ≈ host.u rtol = 1000 * eps(T)
    # The error is a difference of values near 1, so it carries their rounding:
    # compared absolutely, at the type's precision.
    @test abs(dev.records[end].l2 - host.records[end].l2) <= 100 * eps(T)
end

@testset "A time in Float64 does not reach a $DEVICE_T kernel" begin
    # The exact state's callback converts the time on the host: converted in
    # the kernel, a `Float64` time is a `Float64` operand that Metal refuses
    # to compile.
    T = DEVICE_T
    case = DEVICE_CASES[3][2]
    r = wave_errors(T, case; N=8, roots=1, t_end=0.125, backend=DEVICE_BACKEND)
    @test r.l2 < 1e-3
    forest = wave_forest(T, case; N=8, roots=1)
    fs = state_fieldset(forest, 4; backend=DEVICE_BACKEND)
    fill_exact!(fs, case, 0.5)
    fill_coefficients!(TreeWaveGR.coefficient_fieldset(forest; backend=DEVICE_BACKEND),
                       Background{3}(KerrSchild(1.0, 0.6)), 0.5)
    @test dirichlet(case, forest, 0.5) isa CellBoundary
end

@testset "A field set comes back to the host" begin
    T = DEVICE_T
    case = DEVICE_CASES[2][2]
    forest = wave_forest(T, case; N=8, roots=1)
    fs = state_fieldset(forest, 4; backend=DEVICE_BACKEND)
    fill_exact!(fs, case, 0)
    h = TreeWaveGR.hostcopy(fs)
    @test get_backend(h.work) == CPU()
    @test Array(fs.work) == h.work
    dst = state_fieldset(forest, 4)
    @test TreeWaveGR.hostcopy!(dst, h).work == h.work
    @test_throws ArgumentError TreeWaveGR.hostcopy!(state_fieldset(forest, 2), h)
end

if DEVICE == "metal"
    @testset "Float64 is refused where the device has none" begin
        forest = Forest{Float64}((1, 1); N=8)
        @test_throws ArgumentError state_fieldset(forest, 4; backend=DEVICE_BACKEND)
    end
end
