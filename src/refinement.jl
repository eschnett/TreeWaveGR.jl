# The refinement criterion (`CODE.md`, "Refinement and the driver").
#
# A Löhner indicator on `u`, with a noise floor referred to a *global*
# amplitude — TreeWave's lesson: `u` crosses zero, and a floor that scales
# with the local values refines numerical dust (a tail of `3.9e-16, 3.6e-17`
# scored `τ = 0.986` against one). It is evaluated per point by TreeAMR's
# `firing_boxes`, on the field set's backend, and reduced to one flag per
# block, with a hysteresis dead band between the refine and the coarsen
# tolerance.
#
# Beside the indicator, a level *floor*: a ball, possibly moving, inside which
# every block is held at a given level whatever the indicator says. A hole
# case wants resolution near the horizon before anything there has a feature
# for an indicator to see, and TreeExcision's surface needs it.

"""
    lohner(um, u0, up, scale; ε = 1//100) -> τ

The Löhner indicator for three consecutive values along one dimension,

    τ = |u₊ − 2u₀ + u₋| / (|u₊ − u₀| + |u₀ − u₋| + 4ε · scale),

undivided, so `τ ∈ [0, 1]` and it falls as `h` shrinks on smooth data. The
floor is referred to the global amplitude `scale`, not to the local values.
A zero denominator gives zero.
"""
@inline function lohner(um, u0, up, scale; ε=oftype(float(u0), 1 // 100))
    num = abs(up - 2u0 + um)
    den = abs(up - u0) + abs(u0 - um) + 4ε * abs(scale)
    return iszero(den) ? zero(num / one(den)) : num / den
end

"""
    cell_tau(work, idx, b, scale, ε, ::Val{D}) -> τ

The largest [`lohner`](@ref) indicator of `u` (variable 1) along any
dimension at the stored index `idx` of block `b` — what `firing_boxes`
evaluates. Reads one point either side, which the ghosts provide.
"""
@inline function cell_tau(work, idx, b, scale, ε, ::Val{D}) where {D}
    u0 = work[idx..., 1, b]
    τ = zero(u0)
    for d in 1:D
        um = work[Base.setindex(idx, idx[d] - 1, d)..., 1, b]
        up = work[Base.setindex(idx, idx[d] + 1, d)..., 1, b]
        τ = max(τ, lohner(um, u0, up, scale; ε=ε))
    end
    return τ
end

"""
    Refinement(T; refine_tol, coarsen_tol, maxlevel_cap, ε = 1//100,
               floor_level = 0, floor_center, floor_radius, floor_velocity)

The refinement criterion's parameters, in the type `T`:

- `refine_tol`, `coarsen_tol` — a block refines where `τ` exceeds the first
  anywhere, and coarsens only where it stays below the second everywhere;
  `coarsen_tol < refine_tol` is the dead band. No defaults: they are the
  case's to choose.
- `maxlevel_cap` — the highest level the indicator may ask for.
- `ε` — the indicator's noise floor, against the global amplitude of `u`.
- `floor_level`, `floor_center`, `floor_radius`, `floor_velocity` — every
  block reaching into the ball of `floor_radius` about
  `floor_center + floor_velocity·t` is held at level `floor_level` or above
  (`0`, the default, is no floor). The velocity is a moving hole's.
"""
struct Refinement{T,D}
    refine_tol::T
    coarsen_tol::T
    maxlevel_cap::Int
    ε::T
    floor_level::Int
    floor_center::NTuple{D,T}
    floor_radius::T
    floor_velocity::NTuple{D,T}
end

function Refinement(::Type{T}, ::Val{D}; refine_tol, coarsen_tol, maxlevel_cap::Integer,
                    ε=1 // 100, floor_level::Integer=0,
                    floor_center=ntuple(_ -> 0, D), floor_radius=0,
                    floor_velocity=ntuple(_ -> 0, D)) where {T,D}
    coarsen_tol < refine_tol || throw(ArgumentError(
        "coarsen_tol ($coarsen_tol) must lie strictly below refine_tol ($refine_tol): " *
        "the gap between them is the dead band that keeps a block at the threshold " *
        "from refining and coarsening on alternate regrids"))
    floor_level <= maxlevel_cap || throw(ArgumentError(
        "the floor's level $floor_level is above the cap $maxlevel_cap"))
    return Refinement{T,D}(T(refine_tol), T(coarsen_tol), Int(maxlevel_cap), T(ε),
                           Int(floor_level), ntuple(d -> T(floor_center[d]), D),
                           T(floor_radius), ntuple(d -> T(floor_velocity[d]), D))
end

"""
    floor_level(crit::Refinement, forest, key, t) -> Int

The level block `key` is held at or above at time `t`: `crit.floor_level`
if the block reaches into the floor's ball, `0` otherwise. Host-side.
"""
function floor_level(crit::Refinement{T,D}, forest::Forest, key, t) where {T,D}
    crit.floor_level == 0 && return 0
    box = block_extent(T, forest, key)
    s = zero(T)
    for d in 1:D
        c = crit.floor_center[d] + crit.floor_velocity[d] * convert(T, t)
        lo, hi = box[d]
        δ = c < lo ? lo - c : c > hi ? c - hi : zero(T)
        s += δ * δ
    end
    return s <= crit.floor_radius^2 ? crit.floor_level : 0
end

"""
    field_scale(fs) -> scale

The global amplitude of `u`, `max |u|` over the owned points of the
working array: the indicator's reference. Computed once per flagging pass,
outside the predicate.
"""
field_scale(fs::FieldSet{T}) where {T} = mesh_mapreduce(abs, max, zero(T), fs; vars=1:1)

"""
    wave_flags(fs, crit::Refinement, t) -> Vector

One flag per block from the indicator on `fs`'s working array, whose ghosts
must be filled: `(Refine, box)` where `τ > refine_tol` somewhere below the
cap or the block is below its floor, `(Keep, box)` where `τ > coarsen_tol`
somewhere, `Coarsen` where nowhere and the block is above its floor and
level 0, `Keep` otherwise. The boxes are those of the cells above
`coarsen_tol` (the whole block for a floor refinement), which TreeAMR's
buffering dilates.
"""
function wave_flags(fs::FieldSet{T,D}, crit::Refinement{T,D}, t) where {T,D}
    scale = field_scale(fs)
    rtol, ctol, ε = crit.refine_tol, crit.coarsen_tol, crit.ε
    valD = Val(D)
    refires = firing_boxes(fs) do work, idx, b, x
        cell_tau(work, idx, b, scale, ε, valD) > rtol
    end
    boxfires = firing_boxes(fs) do work, idx, b, x
        cell_tau(work, idx, b, scale, ε, valD) > ctol
    end
    whole = ntuple(_ -> 1:(fs.forest.N), D)
    return map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        l = level(k)
        lf = floor_level(crit, fs.forest, k, t)
        nrefine, _ = refires[b]
        nbox, box = boxfires[b]
        if l < lf
            return (Refine, whole)
        elseif nrefine > 0 && l < crit.maxlevel_cap
            return (Refine, box)
        elseif nbox > 0
            return (Keep, box)
        elseif l > max(lf, 0)
            return Coarsen
        else
            return Keep
        end
    end
end

"""
    refinement_buffer(forest, maxlevel_cap, travel) -> cells

The buffer width in cells that covers a feature moving `travel` between one
regrid and the next, at the spacing of level `maxlevel_cap`:
`ceil(travel/h) + 1`. TreeAMR's recruitment reaches one ring of neighbours,
so it must not exceed `N`; this throws naming the constraint rather than
letting `regrid!` refuse it.
"""
function refinement_buffer(forest::Forest, maxlevel_cap::Integer, travel::Real)
    h = spacing(forest, maxlevel_cap)
    cells = ceilint(travel / h) + 1
    cells <= forest.N || throw(ArgumentError(
        "a feature travelling $travel between regrids needs a $cells-cell margin at " *
        "level $maxlevel_cap (h = $h), more than the block width N = $(forest.N): " *
        "regrid more often, or lower the cap"))
    return cells
end
