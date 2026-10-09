# The background spacetime, and the coefficients of the wave equation on it.
#
# `CODE.md`, "The equations" and "Backgrounds". A background is any analytic
# 4-metric from SpacetimeMetrics (or one of the two model metrics below),
# evaluated in `D ≤ 3` spatial dimensions. The wave equation needs only five
# fields of it, the `WaveCoefficients`, and only those reach a kernel.

"""
    Background{D}(m::SpacetimeMetrics.AbstractMetric)

The spacetime `m`, seen by a `D`-dimensional mesh. At `D = 3` this is `m`
itself. At `D < 3` it is the `(D+1)`-dimensional spacetime whose metric is
the leading `(D+1)×(D+1)` block of `m`, evaluated on the slice where the
missing spatial coordinates are zero — a model spacetime, which is what
TreeExcision's two-dimensional model problems are (`CODE.md`,
"Backgrounds"). It is `isbits` whenever `m` is, and can be captured by a
kernel's callback.
"""
struct Background{D,M<:AbstractMetric}
    metric::M
    function Background{D}(m::M) where {D,M<:AbstractMetric}
        D isa Int && 1 <= D <= 3 || throw(ArgumentError(
            "a background is seen by a mesh of 1, 2 or 3 spatial dimensions, " *
            "but D = $D"))
        return new{D,M}(m)
    end
end

Base.ndims(::Background{D}) where {D} = D

# The 4-position `(t, x_1, …, x_D, 0, …)` of a point of the `D`-dimensional
# slice. The zeros are `zero(t)` after promotion, so a dual `t` or `x` gives a
# dual position.
@inline spacetime_point(::Val{1}, t, x) = SVector{4}(promote(t, x[1], zero(t), zero(t))...)
@inline spacetime_point(::Val{2}, t, x) = SVector{4}(promote(t, x[1], x[2], zero(t))...)
@inline spacetime_point(::Val{3}, t, x) = SVector{4}(promote(t, x[1], x[2], x[3])...)

"""
    spacetime_metric(bg::Background{D}, t, x) -> SMatrix{D+1,D+1}

The covariant metric `g_{ab}` of the background at time `t` and position
`x` (an `NTuple` or `SVector` of length `D`), in the coordinates
`(t, x_1, …, x_D)`.
"""
@inline function spacetime_metric(bg::Background{D}, t, x) where {D}
    g = metric(bg.metric, spacetime_point(Val(D), t, x))
    return g[SOneTo(D + 1), SOneTo(D + 1)]
end

"""
    WaveCoefficients{T,D}

The five fields of the background the wave equation reads at one point
(`CODE.md`, "The equations"):

| field | meaning |
|---|---|
| `αsγ` | `α/√γ`, the lapse over the volume element |
| `β` | `β^i`, the shift |
| `divβ` | `∂_i β^i` |
| `A` | `A^{ij} = α√γ γ^{ij}` (symmetric) |
| `divA` | `∂_i A^{ij}`, one entry per `j` |

`2 + 2D + D(D+1)/2` numbers, 14 at `D = 3` — what a sampled coefficient
set stores per point ([`ncoefficients`](@ref)). `isbits`.
"""
struct WaveCoefficients{T,D,L}
    αsγ::T
    β::SVector{D,T}
    divβ::T
    A::SMatrix{D,D,T,L}
    divA::SVector{D,T}
end

"""
    ncoefficients(D) -> Int

The number of independent numbers in a [`WaveCoefficients`](@ref) at
dimension `D`: `2 + 2D + D(D+1)/2`.
"""
ncoefficients(D::Integer) = 2 + 2D + D * (D + 1) ÷ 2

"""
    wave_fields(g::SMatrix{D+1,D+1}) -> (αsγ, β, A)

The undifferentiated coefficients from the covariant metric, by the 3+1
split `γ_{ij} = g_{ij}`, `β^i = γ^{ij} g_{tj}`, `α² = −g_{tt} + β^i g_{ti}`.
Generic in the element type, so a dual metric gives dual fields: that is how
[`wave_coefficients`](@ref) differentiates them.
"""
@inline function wave_fields(g::SMatrix{D1,D1}) where {D1}
    D = D1 - 1
    γ = g[StaticArrays.SUnitRange(2, D1), StaticArrays.SUnitRange(2, D1)]
    βl = g[StaticArrays.SUnitRange(2, D1), 1]
    γu = inv(γ)
    β = γu * βl
    α = sqrt(-g[1, 1] + dot(β, βl))
    sγ = sqrt(det(γ))
    A = (α * sγ) * γu
    # Exactly symmetric: `inv` is not, in the last place, and a sampled set
    # stores one triangle — so the analytic source must see the same matrix.
    return α / sγ, β, (A + A') / 2
end

# The dual-number tag of the coefficients' derivative pass, distinct from
# SpacetimeMetrics' own and from the exact solutions', so that the passes
# nest.
struct CoefficientTag end

# `x` seeded with one partial per spatial direction.
@inline function seed(::Type{Tag}, x::SVector{D,T}) where {Tag,D,T}
    return SVector{D}(ntuple(Val(D)) do i
        ForwardDiff.Dual{Tag}(x[i], ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(T), zero(T)), Val(D))))
    end)
end

"""
    wave_coefficients(bg::Background{D}, t, x) -> WaveCoefficients{T,D}

The coefficients of the wave equation on `bg` at time `t` and position `x`
(an `NTuple` or `SVector` of length `D`), with `T` the element type of `x`.

The divergences `∂_i β^i` and `∂_i A^{ij}` come from one forward-mode pass
through [`wave_fields`](@ref) with `D` partials: the metric is evaluated
once, on dual numbers, and nothing is differentiated twice. Allocation-free
and `isbits` throughout, so it runs inside a kernel — which is what the
analytic coefficient source does at every point of every stage.
"""
@inline function wave_coefficients(bg::Background{D}, t, x) where {D}
    xs = SVector{D}(x)
    T = eltype(xs)
    αsγd, βd, Ad = wave_fields(spacetime_metric(bg, convert(T, t), seed(CoefficientTag, xs)))
    val = ForwardDiff.value
    ∂ = ForwardDiff.partials
    αsγ = val(αsγd)
    β = SVector{D}(ntuple(i -> val(βd[i]), Val(D)))
    A = SMatrix{D,D}(ntuple(k -> val(Ad[k]), Val(D * D)))
    divβ = _trace_partials(βd, Val(D))
    divA = SVector{D}(ntuple(j -> _column_divergence(Ad, j, Val(D)), Val(D)))
    return WaveCoefficients(αsγ, β, divβ, A, divA)
end

# `∑_i ∂_i β^i` and `∑_i ∂_i A^{ij}`, summed from `i = 1` up.
@inline function _trace_partials(βd, ::Val{D}) where {D}
    s = ForwardDiff.partials(βd[1], 1)
    for i in 2:D
        s += ForwardDiff.partials(βd[i], i)
    end
    return s
end
@inline function _column_divergence(Ad, j, ::Val{D}) where {D}
    s = ForwardDiff.partials(Ad[1, j], 1)
    for i in 2:D
        s += ForwardDiff.partials(Ad[i, j], i)
    end
    return s
end

"""
    characteristic_speed(c::WaveCoefficients, d) -> scalar

The largest characteristic speed along grid direction `d`,
`|β^d| + α√(γ^{dd})`, from the stored coefficients: `α² γ^{dd} = (α/√γ)·A^{dd}`.
Both characteristic speeds along `d` are `−β^d ± α√(γ^{dd})` (`CODE.md`, "The
equations"); they have the same sign — the line is outflow at one end —
exactly where `g^{dd} < 0`.
"""
@inline characteristic_speed(c::WaveCoefficients, d::Int) =
    abs(c.β[d]) + sqrt(c.αsγ * c.A[d, d])

# --- packing, for the sampled coefficient set --------------------------------
#
# The order of the stored variables: `αsγ`, `β^1 … β^D`, `∂_iβ^i`, the upper
# triangle of `A` row by row, `∂_iA^{i1} … ∂_iA^{iD}`.

# The `(i, j)` pairs of the upper triangle, row by row.
_triangle(D) = [(i, j) for i in 1:D for j in i:D]

"""
    pack_coefficients(c::WaveCoefficients) -> NTuple{ncoefficients(D)}

The coefficients as the tuple a sampled coefficient set stores at one point,
in the order `αsγ, β, divβ, A (upper triangle, row by row), divA`.
"""
@generated function pack_coefficients(c::WaveCoefficients{T,D}) where {T,D}
    ex = Any[:(c.αsγ)]
    append!(ex, [:(c.β[$i]) for i in 1:D])
    push!(ex, :(c.divβ))
    append!(ex, [:(c.A[$i, $j]) for (i, j) in _triangle(D)])
    append!(ex, [:(c.divA[$j]) for j in 1:D])
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, ex...))
end

"""
    load_coefficients(work, base, sv, ::Val{D}) -> WaveCoefficients

The inverse of [`pack_coefficients`](@ref), reading the stored variables of
one point of a coefficient set's working array at linear index `base`
(variable 1), variables `sv` apart.
"""
@generated function load_coefficients(work, base::Int, sv::Int, ::Val{D}) where {D}
    tri = _triangle(D)
    off(k) = :(base + $(k - 1) * sv)
    k = 1
    αsγ = :(work[$(off(k))])
    β = [:(work[$(off(k + i))]) for i in 1:D]
    k += D + 1
    divβ = :(work[$(off(k))])
    k += 1
    pos = Dict{Tuple{Int,Int},Int}()
    for (n, ij) in enumerate(tri)
        pos[ij] = k + n - 1
    end
    k += length(tri)
    # Column-major, both triangles from the one stored entry.
    Aentries = [:(work[$(off(pos[(min(i, j), max(i, j))]))]) for j in 1:D for i in 1:D]
    divA = [:(work[$(off(k + j - 1))]) for j in 1:D]
    return quote
        $(Expr(:meta, :inline))
        @inbounds WaveCoefficients($αsγ, SVector{$D}($(β...)), $divβ,
                                   SMatrix{$D,$D}($(Aentries...)),
                                   SVector{$D}($(divA...)))
    end
end

# --- model metrics ------------------------------------------------------------

"""
    ConstantShift(β)

Flat spacetime with unit lapse, flat spatial metric and the uniform shift
`β` (a 3-vector), `ds² = −dt² + (dx + β dt)²`: Minkowski in the chart
`x̂ = x + β t`. The shift may be superluminal (`|β| > 1`), and then every
characteristic along `β` leaves through the face `β` points away from — the
classic outflow test, TreeExcision's model problem 1.
"""
struct ConstantShift{T} <: AbstractMetric
    shift::SVector{3,T}
end
ConstantShift(β::AbstractVector) = ConstantShift(SVector{3}(float.(β)))

Base.nameof(m::ConstantShift) = "constant shift (β=$(m.shift))"

function SpacetimeMetrics.metric(m::ConstantShift, x::AbstractVector)
    U = promote_type(eltype(x), eltype(m.shift))
    β = SVector{3,U}(m.shift)
    o, z = one(U), zero(U)
    return SMatrix{4,4,U}(-o + dot(β, β), β[1], β[2], β[3],
                          β[1], o, z, z,
                          β[2], z, o, z,
                          β[3], z, z, o)
end

"""
    River(M)

The "river" metric: unit lapse, flat spatial metric, and the radial shift
`β^i = √(2M/r) x^i/r`, `r = |x|`. At `D = 3` this is Schwarzschild in
Painlevé–Gullstrand coordinates (the horizon at `r = 2M`, inside which both
characteristics along `r` move inward), with the same inverse spatial
metric `g^{ij} = δ^{ij} − (2M/r) n^i n^j` as Kerr–Schild. At `D = 2`, seen
by `Background{2}`, it is TreeExcision's model problem 2: the same
`g^{ij}` with curvature, staircases and every grazing angle, in two
dimensions.

The shift points outward, so that normal observers, moving at `−β`, fall
in (TreeExcision's `notes/questions.md` writes the shift with the opposite
sign convention).
"""
struct River{T} <: AbstractMetric
    mass::T
end

Base.nameof(m::River) = "river (M=$(m.mass))"

function SpacetimeMetrics.metric(m::River, x::AbstractVector)
    U = promote_type(eltype(x), typeof(m.mass))
    t, X, Y, Z = U(x[1]), U(x[2]), U(x[3]), U(x[4])
    r = sqrt(X^2 + Y^2 + Z^2)
    v = sqrt(2 * U(m.mass) / r) / r
    β = SVector{3,U}(v * X, v * Y, v * Z)
    o, z = one(U), zero(U)
    return SMatrix{4,4,U}(-o + dot(β, β), β[1], β[2], β[3],
                          β[1], o, z, z,
                          β[2], z, o, z,
                          β[3], z, z, o)
end
