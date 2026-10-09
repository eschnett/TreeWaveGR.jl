# Where the right-hand side kernel takes the background's coefficients from
# (`CODE.md`, "Coefficient sources").
#
# Two sources behind one seam, chosen by the type of one kernel argument:
#
#   * **sampled** — the coefficients of a static or stationary background,
#     evaluated once per mesh into a coefficient field set (`G = 0`, never
#     ghost-filled) and read back at every stage. The right-hand side then
#     does no metric algebra at all.
#   * **analytic** — the coefficients evaluated in the kernel, at the point
#     and the stage time, by `wave_coefficients`. This is what a
#     time-dependent background (a boosted hole) needs.
#
# The kernel argument is the coefficient set's working array for the first
# and `nothing` for the second. Both evaluate `wave_coefficients` at the same
# position, formed by the same expression TreeAMR's `coordinates` uses, so on
# a background that does not depend on `t` the two give bit-identical
# right-hand sides (`test/evolution_tests.jl`).

"""
    coefficient_fieldset(forest; backend = CPU()) -> FieldSet

A field set for the sampled coefficients over `forest`: vertex-centered like
the state, `G = 0` (it is read only at owned points and never ghost-filled),
with [`ncoefficients`](@ref)`(D)` variables. Over a forest with reflecting
faces or a rotating seam it declares every variable even and unrotated,
which TreeAMR requires of a field set there but which nothing reads: the set
is never exchanged.
"""
function coefficient_fieldset(forest::Forest{D,T}; backend=CPU()) where {D,T}
    n = ncoefficients(D)
    return FieldSet{T}(forest, n; G=0, centering=vertexcentered(D),
                       parity=even_parity(forest, n),
                       rotation=identity_rotation(forest, n), backend=backend)
end

"""
    fill_coefficients!(cfs::FieldSet, bg::Background, t) -> cfs

Sample `bg`'s coefficients at time `t` into the coefficient field set `cfs`,
at every owned point, through `fill_by_coordinates!`.
"""
function fill_coefficients!(cfs::FieldSet{T,D}, bg::Background{D}, t) where {T,D}
    tt = convert(T, t)
    fill_by_coordinates!(AllVariables(x -> pack_coefficients(wave_coefficients(bg, tt, x))),
                         cfs)
    return cfs
end

"""
    point_coefficients(cwork, bg, I, x, t, ::Val{D}) -> WaveCoefficients

The coefficients at owned point `I = (i_1, …, i_D, b)`, position `x` and time
`t`: read from the sampled coefficient set's working array `cwork`, or —
when `cwork` is `nothing` — evaluated from the background `bg`.
"""
@inline function point_coefficients(cwork::AbstractArray, bg, I, x, t, ::Val{D}) where {D}
    st, sv, sb = work_strides(cwork, Val(D))
    base = linear_index(I, ntuple(_ -> 0, Val(D)), st, sb, Val(D))
    return load_coefficients(cwork, base, sv, Val(D))
end
@inline point_coefficients(::Nothing, bg, I, x, t, ::Val{D}) where {D} =
    wave_coefficients(bg, t, x)
