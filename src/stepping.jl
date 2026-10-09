# The time integrator: IMEXRungeKutta's explicit `RK4`, its stage arithmetic
# run by block owner (`CODE.md`, "Time integration").
#
# IMEXRungeKutta rather than OrdinaryDiffEq for the reason TreeHydro and
# TreeGeneralizedHarmonic moved to it: it forms each stage combination as one
# pass over the state split by the partition below, so every block's entries
# are combined on the thread `map_blocks!` runs that block on, and the result
# is bitwise the same at every thread count.

"""
    state_partition(fs::FieldSet, u) -> Vector{UnitRange{Int}} or nothing

The block-ownership partition of the state vector `u` of `fs`, as
IMEXRungeKutta's `partition` keyword takes it: element `c` holds the entries
of the blocks in chunk `c` of TreeAMR's `threadchunks(nblocks(fs))`, padded
with empty ranges where there are fewer blocks than threads. `nothing` for a
state that is not a CPU `Array`, which is IMEXRungeKutta's broadcast path.

Copied from TreeGeneralizedHarmonic; TreeAMR's suite carries the same
helper as a candidate for its own API.
"""
function state_partition(fs::FieldSet{T,D}, u) where {T,D}
    u isa Array || return nothing
    L = fs.forest.N^D * fs.nvars
    length(u) == L * nblocks(fs) || throw(DimensionMismatch(
        "the state vector has $(length(u)) entries but the field set's blocks " *
        "hold $(L * nblocks(fs))"))
    parts = UnitRange{Int}[((first(r) - 1) * L + 1):(last(r) * L)
                           for r in TreeAMR.threadchunks(nblocks(fs))]
    while length(parts) < Threads.nthreads()
        push!(parts, 1:0)
    end
    return parts
end

"""
    wave_integrator(p::WaveProblem, u, t0, t1, nsteps; alias_u0 = false,
                    reuse = nothing, partition = state_partition(p.fs, u))

An IMEXRungeKutta integrator for `nsteps` fixed `RK4` steps from `t0` to `t1`
on [`wave_rhs!`](@ref). `alias_u0 = true` steps `u` itself; `reuse` hands it
an earlier integrator's scratch on the same mesh. Advance it with
`IRK.step!` or `IRK.solve!`; its `u` is then the state and its `t` the time.

The step count is asserted rather than assumed: IMEXRungeKutta derives it as
`⌈(t1 − t0)/dt⌉` to within a few ulp.
"""
function wave_integrator(p::WaveProblem{T}, u, t0, t1, nsteps::Integer;
                         alias_u0::Bool=false, reuse=nothing,
                         partition=state_partition(p.fs, u)) where {T}
    nsteps >= 1 || throw(ArgumentError("nsteps must be at least 1, got $nsteps"))
    t0, t1 = convert(T, t0), convert(T, t1)
    prob = IRK.IMEXProblem(wave_rhs!, nothing, u, (t0, t1), p)
    integ = IRK.init(prob, IRK.RK4(); dt=(t1 - t0) / nsteps, partition=partition,
                     alias_u0=alias_u0, reuse=reuse)
    integ.nsteps == nsteps || throw(ErrorException(
        "IMEXRungeKutta derived $(integ.nsteps) steps from dt = (t1 − t0)/$nsteps"))
    return integ
end

"""
    wave_solve(p::WaveProblem, u, t0, t1, nsteps) -> u1

`nsteps` fixed `RK4` steps from `t0` to `t1`; returns the final state. `u` is
not modified.
"""
function wave_solve(p::WaveProblem, u, t0, t1, nsteps::Integer)
    integ = wave_integrator(p, u, t0, t1, nsteps)
    IRK.solve!(integ)
    return integ.u
end

"""
    wave_steps(p::WaveProblem, t0, t1; cfl = 1/4) -> nsteps

The number of equal steps from `t0` to `t1` that keeps
`dt ≤ cfl · h_min / λ`, with `λ` the [`max_speed`](@ref) at `t0` — at least
1. The speed is measured once; a time-dependent background whose speed
grows over the span needs a smaller `cfl` or a shorter span.
"""
function wave_steps(p::WaveProblem{T}, t0, t1; cfl=1 // 4) where {T}
    λ = max_speed(p, t0)
    dt = convert(T, cfl) * minimum_spacing(T, p.fs.forest) / λ
    return max(1, ceilint((convert(T, t1) - convert(T, t0)) / dt))
end
