# Where the right-hand side takes its stencils from: the seam TreeExcision
# plugs into (`CODE.md`, "Extension points").
#
# The interface is the one TreeGeneralizedHarmonic's excision branch arrived
# at (`StencilProvider`, step X2a there), made `D`-generic, so that a provider
# written for this package's scalar wave answers the same five questions the
# generalized-harmonic kernel asks. The main kernel always uses `Centered`,
# which inlines to the plain stencils: the hot path does not change when a
# different provider exists.

"""
    StencilProvider

What [`wave_rhs_point`](@ref) takes its stencils from. A provider is an
`isbits` value built per point, and it answers five questions, each about
**one variable at one point**, addressed as the stencils address the working
array — by `base`, the linear index of that variable at the point:

| method | returns | the caller scales by |
|---|---|---|
| `d1(S, work, base, d)` | the first derivative along `d`, unit spacing | `1/h` |
| `d2(S, work, base, d)` | the second derivative along `d` | `1/h²` |
| `dmix(S, work, base, i, j)` | `∂_i∂_j`, `i < j`: outer sum along `i`, inner along `j` | `1/h²` |
| `ko(S, work, base, d)` | the Kreiss–Oliger contraction along `d` | `ε/h` |
| `adv(S, β_d, ∂f_d, work, base, d)` | the derivative that multiplies `β^d` | — |

The first four are raw contractions on unit spacing, as
[`derivative_weights`](@ref) and [`dissipation_weights`](@ref) are.

`adv` is asked for the derivative in the advective terms `β^d ∂_d u` and
`β^d ∂_d Π` only, and is handed the shift component `β_d` and the *scaled*
centered derivative `∂f_d` already formed along `d`. It returns a scaled
derivative. [`Centered`](@ref) returns `∂f_d` itself; a lopsided provider
would blend in an upwinded derivative chosen by the sign of `β_d`.
"""
abstract type StencilProvider end

"""
    Centered(T, ::Val{q}, st::NTuple{D,Int}) -> Centered{T,q,D}

The centered stencils of order `q`: the provider of the main kernel. It holds
the working array's per-axis strides ([`work_strides`](@ref)) and nothing
else; the weights are `@generated` constants formed inside each method, so
they are constants wherever the method is compiled and never travel through
the stack of a device that does not inline.
"""
struct Centered{T,q,D} <: StencilProvider
    st::NTuple{D,Int}
end

@inline Centered(::Type{T}, ::Val{q}, st::NTuple{D,Int}) where {T,q,D} =
    Centered{T,q,D}(st)

# Short names because the right-hand side reads as the equation with them; a
# function that calls them must not have a local of the same name.
@inline d1(S::Centered{T,q}, work, base::Int, d::Int) where {T,q} =
    axis_stencil(derivative_weights(T, Val(q), Val(1)), work, base, S.st[d])
@inline d2(S::Centered{T,q}, work, base::Int, d::Int) where {T,q} =
    axis_stencil(derivative_weights(T, Val(q), Val(2)), work, base, S.st[d])
@inline dmix(S::Centered{T,q}, work, base::Int, i::Int, j::Int) where {T,q} =
    mixed_stencil(derivative_weights(T, Val(q), Val(1)), work, base, S.st[i],
                  S.st[j])
@inline ko(S::Centered{T,q}, work, base::Int, d::Int) where {T,q} =
    axis_stencil(dissipation_weights(T, dissipation_rank(Val(q))), work, base,
                 S.st[d])
@inline adv(S::Centered, β_d, ∂f_d, work, base::Int, d::Int) = ∂f_d
