# The faces of the domain (`CODE.md`, "Boundaries").
#
# Periodic faces, reflecting faces and the rotating seam are properties of
# TreeAMR's forest, and its ghost exchange fills them itself; what it needs
# from an application is how the variables behave under a mirror and under a
# quarter turn. `u` and `Π` are scalars, so both are even under every mirror
# and turn into themselves — the simplest declaration there is.
#
# Every other face is an outer face, filled by a hook. This package has one:
# Dirichlet data from the case's exact solution at the current time, as a
# TreeAMR `CellBoundary`, which runs as a kernel on any backend. A radiative
# condition would need a hook that reads the interior, which TreeAMR offers
# only on the host; none is needed for what this package is for, since every
# case it runs has an exact solution (`CODE.md`, "Possible extensions").

# Whether the forest has a reflecting face, or a rotating seam.
_reflects(forest) = any(r -> r[1] || r[2], forest.reflecting)
seam_dims(forest::Forest) =
    forest.rotating[1] == 0 ? nothing : (Int(forest.rotating[1]), Int(forest.rotating[2]))

"""
    even_parity(forest, nvars) -> Vector or nothing

`EvenParity` for every variable, or `nothing` over a forest without a
reflecting face: the `parity` of the state (`u` and `Π` are scalars) and of
every auxiliary field set.
"""
even_parity(forest::Forest{D}, nvars::Integer) where {D} =
    _reflects(forest) ? fill(ntuple(_ -> EvenParity, Val(D)), Int(nvars)) : nothing

"""
    identity_rotation(forest, nvars) -> Vector{Int} or nothing

`1:nvars`, every variable turning into itself, or `nothing` over a forest
without a rotating seam: the `rotation` of the state and of every auxiliary
field set.
"""
identity_rotation(forest, nvars::Integer) =
    seam_dims(forest) === nothing ? nothing : collect(1:Int(nvars))

"""
    has_outer_face(forest) -> Bool

Whether the forest has a face that is neither periodic nor reflecting nor on
the rotating seam — one the Dirichlet hook fills.
"""
function has_outer_face(forest::Forest{D}) where {D}
    seam = seam_dims(forest)
    for d in 1:D
        forest.periodic[d] && continue
        lo, hi = forest.reflecting[d]
        onseam = seam !== nothing && d in seam
        (lo || onseam) || return true
        hi || return true
    end
    return false
end

"""
    dirichlet(case::WaveCase, forest, t) -> CellBoundary or nothing

The Dirichlet hook at time `t`: every outer ghost point set to the case's
exact state there, through [`exact_state`](@ref). `nothing` when the
forest has no outer face. Rebuilt at every right-hand-side evaluation with
that evaluation's `t`; it captures only `isbits` values, so it is a kernel
argument on any backend. The case is retyped to the forest's type first.

It goes to three places, as in TreeGeneralizedHarmonic: the ghost fill of
every right-hand side, `regrid!`, and `adapt_to_initial_data!`.
"""
dirichlet(case, forest::Forest{D,T}, t) where {D,T} =
    _dirichlet(retype(T, case), forest, t)

# The hook for a case already in the forest's type — what `WaveProblem` and
# `evolve!` hold — without `retype`'s reflection at every right-hand side.
function _dirichlet(case, forest::Forest{D,T}, t) where {D,T}
    has_outer_face(forest) || return nothing
    sol, bg = case.solution, case.background
    tt = convert(T, t)
    return CellBoundary(AllVariables((x, δ) -> exact_state(sol, bg, tt, x)))
end
