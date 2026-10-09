# The cases: a background, an exact solution on it, and a domain
# (`CODE.md`, "Cases").

"""
    WaveCase(background, solution; extents, periodic, reflecting, rotating,
             ε = 0)

What one run is: a [`Background`](@ref), an [`ExactSolution`](@ref) on it
(for the initial data, the Dirichlet data at every outer face, and the
errors), the domain — `extents`, one `(lo, hi)` per dimension, and which
faces are periodic or reflecting and which pair of dimensions is a rotating
seam, as TreeAMR's `Forest` takes them — and the Kreiss–Oliger strength `ε`.
`isbits` whenever the background and solution are, so that it is captured
by the Dirichlet hook.
"""
struct WaveCase{D,B<:Background{D},S<:ExactSolution,T}
    background::B
    solution::S
    extents::NTuple{D,Tuple{T,T}}
    periodic::NTuple{D,Bool}
    reflecting::NTuple{D,Tuple{Bool,Bool}}
    rotating::NTuple{2,Int}
    ε::T
end

function WaveCase(background::Background{D}, solution::ExactSolution;
                  extents, periodic=ntuple(_ -> false, D),
                  reflecting=ntuple(_ -> (false, false), D), rotating=nothing,
                  ε=0) where {D}
    check_solution(solution, background)
    check_slice(background, extents)
    T = float(promote_type(map(e -> promote_type(typeof.(e)...), extents)..., typeof(ε)))
    ext = ntuple(d -> (T(extents[d][1]), T(extents[d][2])), D)
    rot = rotating === nothing ? (0, 0) : (Int(rotating[1]), Int(rotating[2]))
    return WaveCase{D,typeof(background),typeof(solution),T}(
        background, solution, ext, Tuple(periodic), Tuple(reflecting), rot, T(ε))
end

Base.ndims(::WaveCase{D}) where {D} = D

"""
    check_slice(bg::Background{D}, extents)

Throw an `ArgumentError` if, at the center of the domain, the background's
metric couples the `D` evolved dimensions to the ones the slice drops — a
`g_{tk}` or `g_{ik}` with `k > D` that is not zero. The leading block of such
a metric is still a metric, but it is not the spacetime an exact solution of
the full one solves the wave equation on (a constant shift with a component
along a dropped axis is the simplest case). Host-side, once per case; a
necessary condition, not a sufficient one, since it does not see a
dependence on the dropped coordinates.
"""
function check_slice(bg::Background{D}, extents) where {D}
    D == 3 && return nothing
    x = ntuple(d -> (float(extents[d][1]) + float(extents[d][2])) / 2, D)
    g = SpacetimeMetrics.metric(bg.metric, spacetime_point(Val(D), zero(x[1]), x))
    for a in 1:(D + 1), k in (D + 2):4
        iszero(g[a, k]) || throw(ArgumentError(
            "the background couples the evolved coordinates to a dropped one " *
            "(g[$a, $k] = $(g[a, k]) at the domain's center), so its slice at " *
            "D = $D is not a reduction of it"))
    end
    return nothing
end

"""
    retype(T, x)

`x` with every floating-point number in it converted to `T`: a number, a
static array, a tuple, a `WaveCase`, a `Background`, an `ExactSolution` or a
SpacetimeMetrics metric — the last two rebuilt field by field through their
unparameterized constructors, which every type here has.

A case is written in whatever literals are convenient — `KerrSchild(1.0,
0.6)` is `Float64` — and a kernel in `Float32` that reads a `Float64`
parameter computes in `Float64`, which a device without hardware `Float64`
refuses to compile. So every entry point that builds a run (`WaveProblem`,
`fill_exact!`, `wave_errors`, `evolve!`) retypes the case to the run's
type once, host-side.
"""
retype(::Type{T}, x::AbstractFloat) where {T} = convert(T, x)
retype(::Type{T}, x::Union{Integer,Rational,Symbol,Nothing}) where {T} =
    x isa Rational ? convert(T, x) : x
retype(::Type{T}, x::Tuple) where {T} = map(y -> retype(T, y), x)
retype(::Type{T}, x::StaticArray) where {T} =
    eltype(x) <: AbstractFloat ? similar_type(x, T)(map(y -> retype(T, y), Tuple(x))) : x
retype(::Type{T}, bg::Background{D}) where {T,D} = Background{D}(retype(T, bg.metric))
function retype(::Type{T}, x::Union{AbstractMetric,ExactSolution}) where {T}
    fields = map(f -> retype(T, getfield(x, f)), fieldnames(typeof(x)))
    return isempty(fields) ? x : typeof(x).name.wrapper(fields...)
end
function retype(::Type{T}, c::WaveCase{D,B,S,T}) where {D,B,S,T}
    # Already in `T` throughout? The common case, and cheap to recognise.
    retyped_already(c.background.metric, T) && retyped_already(c.solution, T) && return c
    return _retype_case(T, c)
end
retype(::Type{T}, c::WaveCase) where {T} = _retype_case(T, c)

_retype_case(::Type{T}, c::WaveCase) where {T} =
    WaveCase(retype(T, c.background), retype(T, c.solution);
             extents=retype(T, c.extents), periodic=c.periodic, reflecting=c.reflecting,
             rotating=c.rotating[1] == 0 ? nothing : c.rotating, ε=convert(T, c.ε))

# Whether every floating-point number reachable from `x`'s fields is a `T`.
function retyped_already(x, ::Type{T}) where {T}
    x isa AbstractFloat && return x isa T
    x isa StaticArray && return !(eltype(x) <: AbstractFloat) || eltype(x) === T
    x isa Union{Integer,Symbol,Nothing} && return true
    x isa Tuple && return all(y -> retyped_already(y, T), x)
    return all(f -> retyped_already(getfield(x, f), T), fieldnames(typeof(x)))
end

"""
    with_dissipation(case, ε) -> WaveCase

`case` with the Kreiss–Oliger strength `ε`.
"""
with_dissipation(c::WaveCase, ε) =
    WaveCase(c.background, c.solution; extents=c.extents, periodic=c.periodic,
             reflecting=c.reflecting,
             rotating=c.rotating[1] == 0 ? nothing : c.rotating, ε=ε)

"""
    wave_forest(T, case; N, roots, refined = false) -> Forest{T}

The case's domain as a forest of `roots` (a number per dimension, or one
for all) blocks of `N^D` points. Blocks are cubes, so `roots` must have the
aspect ratio of the extents. With `refined = true` the first root is refined
once and the forest balanced: a frozen two-level hierarchy for the
interface tests.
"""
function wave_forest(::Type{T}, case::WaveCase{D}; N::Integer, roots,
                     refined::Bool=false) where {T,D}
    r = roots isa Integer ? ntuple(_ -> Int(roots), D) : Tuple(roots)
    forest = Forest{T}(r; N=N, periodic=case.periodic, reflecting=case.reflecting,
                       rotating=case.rotating[1] == 0 ? nothing : case.rotating,
                       extents=case.extents)
    if refined
        refine!(forest, [forest.leaves[1]])
        balance!(forest)
    end
    return forest
end

"""
    state_fieldset(forest, q; backend = CPU()) -> FieldSet

The state over `forest` at order `q`: two variables, `u` and `Π`,
vertex-centered, with ghost width `G = q/2 + 1`.
"""
function state_fieldset(forest::Forest{D,T}, q::Integer; backend=CPU()) where {D,T}
    return FieldSet{T}(forest, 2; G=q ÷ 2 + 1, centering=vertexcentered(D),
                       parity=even_parity(forest, 2), rotation=identity_rotation(forest, 2),
                       backend=backend)
end

"""
    wave_operators(q) -> Operators

The inter-level operators for order `q`: point-value prolongation and
restriction of order `q + 2`, two above the differencing, which a second
derivative needs to keep order `q` across a coarse-fine interface.
"""
wave_operators(q::Integer) = Operators(prolongation=q + 2, restriction=q + 2)

"""
    exact_callback(case, t) -> AllVariables

The case's exact state at time `t`, as the `AllVariables` callback
`fill_by_coordinates!` and `adapt_to_initial_data!` take.
"""
function exact_callback(case::WaveCase{D,B,S,T}, t) where {D,B,S,T}
    sol, bg = case.solution, case.background
    return AllVariables(x -> exact_state(sol, bg, convert(eltype(x), t), x))
end

"""
    fill_exact!(fs, case, t) -> fs

Every owned point of `fs` set to the case's exact state at time `t`.
"""
fill_exact!(fs::FieldSet{T}, case::WaveCase, t) where {T} =
    fill_by_coordinates!(exact_callback(retype(T, case), t), fs)

"""
    exact_statevector(fs, case, t) -> u

The case's exact state at time `t` as a state vector of `fs`. Overwrites
`fs`'s working array.
"""
function exact_statevector(fs::FieldSet, case::WaveCase, t)
    fill_exact!(fs, case, t)
    u = statevector(fs)
    gather!(u, fs)
    return u
end

"""
    wave_errors(T, case; N, roots, q = 4, t_end, cfl = 1/4, refined = false,
                coefficients = :auto, backend = CPU()) -> NamedTuple

One fixed-step run of `case` from its exact state at `t = 0` to `t_end` on a
uniform (or, with `refined`, frozen two-level) mesh, and its error against
the exact state there: `(; l2, linf, h, nsteps, nblocks)`, with `l2` and
`linf` the volume-weighted norms over both variables. What every
convergence test is built from.
"""
function wave_errors(::Type{T}, case::WaveCase; N::Integer, roots, q::Integer=4,
                     t_end, cfl=1 // 4, refined::Bool=false, coefficients=:auto,
                     backend=CPU()) where {T}
    case = retype(T, case)
    forest = wave_forest(T, case; N=N, roots=roots, refined=refined)
    fs = state_fieldset(forest, q; backend=backend)
    p = WaveProblem(fs, GhostSchedule(fs, wave_operators(q)), case; q=q,
                    coefficients=coefficients)
    u0 = exact_statevector(fs, case, 0)
    nsteps = wave_steps(p, 0, t_end; cfl=cfl)
    u1 = wave_solve(p, u0, 0, t_end, nsteps)
    ex = exact_statevector(fs, case, t_end)
    err = u1 .- ex
    return (; l2=volume_weighted_norm(fs, err), linf=volume_weighted_norm(fs, err; p=Inf),
            h=minimum_spacing(T, forest), nsteps, nblocks=nblocks(fs))
end
