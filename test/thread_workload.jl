# The workload behind the thread-count independence test.
#
# Run as a standalone script —
#
#     julia -t N --project=. test/thread_workload.jl
#
# — it prints a digest of every state vector and every reduction a short run
# produces. Two runs at different thread counts must print the same lines,
# character for character: every parallel loop in TreeAMR writes its own
# slot, every combination of partials happens in block order, IMEXRungeKutta
# combines stages by block owner, and this package adds no loop of its own.
#
# Nothing outside `Base` and the package's own dependencies, no randomness,
# no timing; `repr` round-trips a `Float64`, so a difference in the last bit
# shows.

using SpacetimeMetrics
using TreeAMR
using TreeWaveGR

digest(u::Vector{<:Real}) = string(hash(u); base=16, pad=16)
digest(s::AbstractString) = string(hash(s); base=16, pad=16)

function thread_digests()
    lines = String[]
    # A two-level mesh on a time-dependent background, analytic
    # coefficients, dissipation on, periodic.
    case = WaveCase(Background{2}(GaugeWave(0.1, 1.0)), PlaneWave(1.0, [2π, 2π, 0]);
                    extents=((0.0, 1.0), (0.0, 1.0)), periodic=(true, true), ε=0.25)
    forest = wave_forest(Float64, case; N=8, roots=2, refined=true)
    fs = state_fieldset(forest, 4)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(4)), case; q=4)
    u = exact_statevector(fs, case, 0)
    n = wave_steps(p, 0, 1 // 4)
    u = wave_solve(p, u, 0, 1 // 4, n)
    push!(lines, "gauge wave nsteps $n state $(digest(u))")
    push!(lines, "gauge wave l2 $(repr(volume_weighted_norm(fs, u))) " *
                 "speed $(repr(max_speed(p, 1 // 4)))")
    # A hole with Dirichlet faces and sampled coefficients.
    case = WaveCase(Background{3}(KerrSchild(1.0, 0.6)), StaticHole(1.0, 0.6);
                    extents=((2.5, 4.5), (2.5, 4.5), (-1.0, 1.0)), ε=0.25)
    forest = wave_forest(Float64, case; N=8, roots=2, refined=true)
    fs = state_fieldset(forest, 4)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(4)), case; q=4)
    u = exact_statevector(fs, case, 0)
    n = wave_steps(p, 0, 1 // 8)
    u = wave_solve(p, u, 0, 1 // 8, n)
    push!(lines, "hole nsteps $n state $(digest(u))")
    push!(lines, "hole l2 $(repr(volume_weighted_norm(fs, u))) " *
                 "speed $(repr(max_speed(p, 1 // 8)))")
    # The driver with its regrids: the initial-data cycle, `firing_boxes` and
    # the indicator's global scale, regrids that move data, and the records'
    # norms.
    case = WaveCase(Background{2}(Minkowski()), PlanePulse(1.0, [1.0, 0.5, 0], 0.08, 0.3);
                    extents=((0.0, 1.0), (0.0, 1.0)), ε=0.25)
    crit = Refinement(Float64, Val(2); refine_tol=0.05, coarsen_tol=0.01, maxlevel_cap=2)
    r = evolve!(Float64, case; N=8, roots=4, t_end=1 // 16, chunk=1 // 32,
                refinement=crit)
    push!(lines, "pulse passes $(r.passes) regrids $(r.nregrids) nsteps $(r.nsteps) " *
                 "blocks $(length(r.forest.leaves)) leaves $(digest(repr(r.forest.leaves)))")
    push!(lines, "pulse state $(digest(r.u)) records $(digest(repr(r.records)))")
    return lines
end

if abspath(PROGRAM_FILE) == @__FILE__
    foreach(println, thread_digests())
end
