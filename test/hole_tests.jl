# The black-hole cases: TreeExcision's success ladder (its `GOAL.md`) for the
# scalar wave — a = 0, 0.6, 0.9; v = 0.3, 0.6; a = 0.6 with v = 0.6 — on a box
# outside the horizon, and the symmetric domains a hole run uses.

# The box: `x ∈ [3, 5]`, clear of every horizon here (`r₊ ≤ 2`) with the
# ghosts included; the moving holes move away from it (`boost` by `v` moves
# the hole at `−v`).
const HOLE_BOX = ((3.0, 5.0), (-1.0, 1.0), (-1.0, 1.0))

const HOLE_CASES = [
    ("a=0", KerrSchild(1.0, 0.0), StaticHole(1.0, 0.0)),
    ("a=0.6", KerrSchild(1.0, 0.6), StaticHole(1.0, 0.6)),
    ("a=0.9", KerrSchild(1.0, 0.9), StaticHole(1.0, 0.9)),
    ("v=0.3", boost(KerrSchild(1.0, 0.0), [0.3, 0, 0]), StaticHole(1.0, 0.0)),
    ("v=0.6", boost(KerrSchild(1.0, 0.0), [0.6, 0, 0]), StaticHole(1.0, 0.0)),
    ("a=0.6, v=0.6", boost(KerrSchild(1.0, 0.6), [0.6, 0, 0]), StaticHole(1.0, 0.6)),
]

@testset "The right-hand side converges at order q on the hole: $name, q=$q" for (name, m, sol) in
                                                                                 HOLE_CASES,
                                                                             q in (2, 4)
    case = WaveCase(Background{3}(m), sol; extents=HOLE_BOX)
    rs = [rhs_error(Float64, case; N=N, roots=1, q=q) for N in (8, 12, 16)]
    rate = convergence_rate([r.h for r in rs], [r.err for r in rs])
    @info "hole rhs" name q rate
    @test rate >= q - 1 // 4
end

@testset "A run on the hole converges: $name" for (name, m, sol) in HOLE_CASES
    # At order q = 4, to t = 1/4. Dirichlet data on every ghost over-specify
    # the outgoing modes, so a run converges at q − 1 asymptotically
    # (`CODE.md`, "Boundaries"); measured at N = 16, 24, 32: 3.6 on the static
    # holes, 3.1 at v = 0.3, 2.7 at v = 0.6.
    case = WaveCase(Background{3}(m), sol; extents=HOLE_BOX)
    rs = [wave_errors(Float64, case; N=N, roots=1, q=4, t_end=1 // 4) for N in (16, 24, 32)]
    errs = [r.l2 for r in rs]
    rate = convergence_rate([r.h for r in rs], errs)
    @info "hole run" name errs rate
    @test issorted(errs; rev=true)
    @test rate >= 5 // 2
    @test errs[end] < 2e-6
end

@testset "A run on the river converges: D=2" begin
    case = WaveCase(Background{2}(River(1.0)), StaticRiver(1.0);
                    extents=((3.0, 5.0), (-1.0, 1.0)))
    rs = [wave_errors(Float64, case; N=N, roots=1, q=4, t_end=1 // 2) for N in (16, 32, 64)]
    rate = convergence_rate([r.h for r in rs], [r.l2 for r in rs])
    @info "river run" rate
    @test rate >= 3
end

# Run `case` on `forest` to `t_end` and return the owned states keyed by
# their position in units of the spacing (the two forests of a comparison
# have the same spacing but different origins).
function keyed_run(case, forest; q=4, t_end=1 // 8, nsteps=4)
    fs = state_fieldset(forest, q)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(q)), case; q=q)
    u = wave_solve(p, exact_statevector(fs, case, 0), 0, t_end, nsteps)
    scatter!(fs, u)
    h = minimum_spacing(forest)
    states = Dict{NTuple{3,Int},NTuple{2,Float64}}()
    for b in 1:nblocks(fs)
        block = interiorview(fs, b)
        for I in CartesianIndices(ntuple(_ -> forest.N, 3))
            x = coordinates(fs, b, Tuple(I) .+ fs.G)
            states[round.(Int, x ./ h)] = (block[Tuple(I)..., 1], block[Tuple(I)..., 2])
        end
    end
    return states
end

# The largest relative difference of the symmetric run from the full one,
# over the symmetric run's points.
#
# Not rounding alone: a vertex-centered block evolves its low boundary plane
# and the Dirichlet hook fills the high one, so the full box is itself
# symmetric only to its truncation error (6e-6 here), and that asymmetry
# reaches the compared half at 1e-12 in four steps — every RK4 stage widens
# the numerical domain of dependence by a stencil. A wrong parity, or a seam
# that turned the wrong way, differs at the size of the solution.
function symmetric_difference(sym, full)
    scale = maximum(v -> max(abs(v[1]), abs(v[2])), values(full))
    return maximum(keys(sym)) do k
        maximum(abs.(sym[k] .- full[k]))
    end / scale
end

@testset "A reflecting face is the mirrored box: a=0.6 with spin along z" begin
    # Kerr–Schild with the spin along `z` is symmetric under `z → −z`, and
    # `u` and `Π` are scalars: the half box above a reflecting face at `z = 0`
    # is the full box's upper half.
    sol = StaticHole(1.0, 0.6)
    bg = Background{3}(KerrSchild(1.0, 0.6))
    half = WaveCase(bg, sol; extents=((3.0, 5.0), (-1.0, 1.0), (0.0, 1.0)),
                    reflecting=((false, false), (false, false), (true, false)), ε=0.25)
    full = WaveCase(bg, sol; extents=((3.0, 5.0), (-1.0, 1.0), (-1.0, 1.0)), ε=0.25)
    fh = wave_forest(Float64, half; N=8, roots=(2, 2, 1))
    ff = wave_forest(Float64, full; N=8, roots=(2, 2, 2))
    sh, sf = keyed_run(half, fh), keyed_run(full, ff)
    @test all(k -> haskey(sf, k), keys(sh))
    @test symmetric_difference(sh, sf) < 1e-10
end

@testset "A rotating seam is the quarter-turned box: a=0.6 with spin along z" begin
    # Kerr–Schild with the spin along `z` is axisymmetric about `z`: the
    # quadrant `x, y ≥ 0` glued to itself by a quarter turn is the full box's
    # quadrant. Above the hole, so that the axis is regular.
    sol = StaticHole(1.0, 0.6)
    bg = Background{3}(KerrSchild(1.0, 0.6))
    quadrant = WaveCase(bg, sol; extents=((0.0, 2.0), (0.0, 2.0), (3.0, 5.0)),
                        rotating=(1, 2), ε=0.25)
    full = WaveCase(bg, sol; extents=((-2.0, 2.0), (-2.0, 2.0), (3.0, 5.0)), ε=0.25)
    fq = wave_forest(Float64, quadrant; N=8, roots=(1, 1, 1))
    ff = wave_forest(Float64, full; N=8, roots=(2, 2, 1))
    sq, sf = keyed_run(quadrant, fq), keyed_run(full, ff)
    @test all(k -> haskey(sf, k), keys(sq))
    @test symmetric_difference(sq, sf) < 1e-10
end
