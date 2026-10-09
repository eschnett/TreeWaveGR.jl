# The driver: a run in chunks, with a regrid between chunks
# (`CODE.md`, "Refinement and the driver").
#
# Each chunk is a fixed number of equal RK4 steps on one mesh, sized by the
# characteristic speed at the chunk's start. Between chunks: a record of the
# state and its error against the exact solution, the observer, a regrid,
# and a checkpoint. This is TreeGeneralizedHarmonic's and TreeHydro's loop,
# with nothing of their physics.

"""
    chunk_end(c, chunk, t_end) -> t

The time at the end of chunk `c` of a run to `t_end` in chunks of `chunk`:
`c · chunk`, or `t_end` for the last. Computed rather than accumulated, so
that a restart after chunk `c` starts at the same bits.
"""
chunk_end(c::Integer, chunk::T, t_end::T) where {T} =
    c >= ceilint(t_end / chunk) ? t_end : c * chunk

"""
    evolve!(T, case; N, roots, q = 4, t_end, chunk = t_end, cfl = 1/4,
            refinement = nothing, buffer = :auto, maxpasses = 10,
            coefficients = :auto, backend = CPU(), observer = nothing,
            checkpoint = nothing, checkpoint_every = 1, restart = nothing,
            types = ()) -> NamedTuple

Run `case` from its exact state at `t = 0` to `t_end`, in chunks of
`chunk`, on a forest of `roots` blocks of `N^D` points at order `q`.

- `refinement` — a [`Refinement`](@ref), or `nothing` for a fixed mesh. With
  one, the initial mesh is adapted to the exact initial data
  (`adapt_to_initial_data!`, at most `maxpasses` passes), and the mesh is
  regridded between chunks. `buffer` is the regrid's margin in cells;
  `:auto` derives it from the distance the fastest characteristic travels in
  one chunk ([`refinement_buffer`](@ref)).
- `observer(p, t, u, row)` is called after every chunk, with the problem,
  the time, the state vector and the chunk's record.
- `checkpoint` — a path prefix; a checkpoint is written after every
  `checkpoint_every` chunks and after the last, at
  [`checkpoint_path`](@ref)`(checkpoint, c)`, after the regrid.
- `restart` — the path of such a checkpoint: the run continues from it, as
  if it had not stopped. The case and the keywords must be the run's own.

Returns `(; records, forest, fs, u, t, nsteps, nregrids, passes)`.
`records` holds one `NamedTuple` of `Float64`s and counts per chunk:
`chunk`, `t`, `nsteps`, `nblocks`, `maxlevel`, `l2`/`linf` (the state's
error against the exact solution) and `norm` (the state's `L2` norm).
"""
function evolve!(::Type{T}, case::WaveCase{D}; N::Integer, roots, q::Integer=4, t_end,
                 chunk=t_end, cfl=1 // 4, refinement=nothing, buffer=:auto,
                 maxpasses::Integer=10, coefficients::Symbol=:auto, backend=CPU(),
                 observer=nothing, checkpoint=nothing, checkpoint_every::Integer=1,
                 restart=nothing, types=()) where {T,D}
    case = retype(T, case)
    t_end, chunk = convert(T, t_end), convert(T, chunk)
    chunk > 0 || throw(ArgumentError("chunk must be positive, got $chunk"))
    nchunks = ceilint(t_end / chunk)
    ops = wave_operators(q)

    nsteps_total, nregrids, passes = 0, 0, 0
    if restart === nothing
        forest = wave_forest(T, case; N=N, roots=roots)
        fs = state_fieldset(forest, q; backend=backend)
        c0 = 0
        if refinement === nothing
            schedule = GhostSchedule(fs, ops)
            fill_exact!(fs, case, zero(T))
        else
            buf = regrid_buffer(T, buffer, case, refinement, chunk; N=N, roots=roots, q=q,
                                coefficients=coefficients)
            schedule, passes, _ = adapt_to_initial_data!(
                fs, ops; initial=exact_callback(case, zero(T)),
                flags=f -> wave_flags(f, refinement, zero(T)), buffer=buf,
                maxpasses=maxpasses, boundary=dirichlet(case, forest, zero(T)))
        end
        u = statevector(fs)
        gather!(u, fs)
    else
        run = load_run(restart; backend=backend, types=types)
        run.data.q == q || throw(ArgumentError(
            "$restart was written at order q = $(run.data.q), this run is at q = $q"))
        run.forest.N == N || throw(ArgumentError(
            "$restart has blocks of N = $(run.forest.N), this run has N = $N"))
        forest, fs, u = run.forest, run.fs, run.u
        eltype(u) === T || throw(ArgumentError(
            "$restart holds a state of $(eltype(u)), this run is in $T"))
        schedule = GhostSchedule(fs, ops)
        c0 = run.data.chunk
        nsteps_total, nregrids = run.data.nsteps, run.data.nregrids
    end
    buf = refinement === nothing ? 0 :
          regrid_buffer(T, buffer, case, refinement, chunk; N=N, roots=roots, q=q,
                        coefficients=coefficients)

    records = NamedTuple[]
    t = c0 == 0 ? zero(T) : chunk_end(c0, chunk, t_end)
    p = nothing
    integ = nothing
    for c in (c0 + 1):nchunks
        stop = chunk_end(c, chunk, t_end)
        if p === nothing
            p = WaveProblem(fs, schedule, case; q=q, coefficients=coefficients)
            integ = nothing
        end
        nsteps = wave_steps(p, t, stop; cfl=cfl)
        # The previous chunk's scratch, while the mesh is the one it was
        # allocated for (`integ` is reset with `p` after a regrid).
        integ = wave_integrator(p, u, t, stop, nsteps; alias_u0=true, reuse=integ)
        IRK.solve!(integ)
        u = integ.u
        t = stop
        nsteps_total += nsteps
        row = chunk_record(p, u, t, c, nsteps)
        push!(records, row)
        observer === nothing || observer(p, t, u, row)

        if refinement !== nothing && c < nchunks
            boundary = dirichlet(case, forest, t)
            scatter!(fs, u)
            fill_ghosts!(fs, schedule; boundary=boundary)
            flags = wave_flags(fs, refinement, t)
            if regrid!(forest, fs => schedule; flags=flags, buffer=buf, boundary=boundary)
                schedule = GhostSchedule(fs, ops)
                u = statevector(fs)
                gather!(u, fs)
                p = nothing
                nregrids += 1
            end
        end
        if checkpoint !== nothing && (c % checkpoint_every == 0 || c == nchunks)
            save_run(checkpoint_path(checkpoint, c), forest, fs, u; chunk=c,
                     nsteps=nsteps_total, nregrids=nregrids, q=q)
        end
    end
    return (; records, forest, fs, u, t, nsteps=nsteps_total, nregrids, passes)
end

# The regrid margin: `buffer` itself, or from the distance the fastest
# characteristic travels in one chunk, measured on the unrefined forest at
# `t = 0` — a function of the run's arguments alone, so that a restart
# derives the same margin as the run it continues.
function regrid_buffer(::Type{T}, buffer, case, refinement, chunk; N, roots, q,
                       coefficients) where {T}
    buffer === :auto || return Int(buffer)
    forest = wave_forest(T, case; N=N, roots=roots)
    fs = state_fieldset(forest, q)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(q)), case; q=q,
                    coefficients=coefficients)
    travel = max_speed(p, 0) * chunk
    return refinement_buffer(forest, refinement.maxlevel_cap, travel)
end

# One chunk's record: the state's error against the exact solution and its
# norm, as `Float64`s, and the mesh's size. Overwrites the working array,
# which the next right-hand side or regrid refreshes from `u`.
function chunk_record(p::WaveProblem{T}, u, t, c, nsteps) where {T}
    fs = p.fs
    err = u .- exact_statevector(fs, p.case, t)
    return (; chunk=c, t=tofloat64(t), nsteps, nblocks=nblocks(fs),
            maxlevel=maxlevel(fs.forest), l2=tofloat64(volume_weighted_norm(fs, err)),
            linf=tofloat64(volume_weighted_norm(fs, err; p=Inf)),
            norm=tofloat64(volume_weighted_norm(fs, u)))
end
