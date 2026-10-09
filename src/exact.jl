# Exact solutions of the wave equation on the backgrounds, for initial data,
# Dirichlet boundary data and error measurement (`CODE.md`, "Exact
# solutions").
#
# A scalar transforms trivially under a change of chart: if `f` solves the
# wave equation in the chart `x̂`, then `u(x) = f(x̂(x))` solves it in the
# chart `x`. SpacetimeMetrics builds its test backgrounds exactly so — a
# boosted, translated or rotated Kerr–Schild hole, flat space in a gauge-wave,
# sine-shift, skewed-time or moving-grid chart — as the pullback of an inner
# metric through a chart map, and each type's `metric` method spells that map
# out internally. `inner_coordinates` spells it out once more, as a function
# of its own, for every such type; the pullback test
# (`test/exact_tests.jl`) holds each one to the metric it claims to pull back.
#
# The solutions themselves are a handful in three base charts: plane waves
# and a polynomial wave on Minkowski, the static monopole on Kerr–Schild, and
# the static monopole on the river metric. Every combination of a solution
# with a chain of wrappers over its base is an exact solution on that
# background — including the superluminal ones, and the moving holes.

"""
    inner_coordinates(m::AbstractMetric, x::SVector{4}) -> SVector{4}

The point `x̂` of `m`'s **base chart** — the innermost metric of a chain of
SpacetimeMetrics wrappers — that the point `x` of `m`'s chart maps to. The
identity for a base metric (`Minkowski`, `KerrSchild`, [`River`](@ref)).
[`ConstantShift`](@ref) is Minkowski in the chart `x̂ = x + β t`, and maps
there.

Generic in the element type, so that a dual `x` differentiates through it.
"""
function inner_coordinates end

inner_coordinates(::Minkowski, x::SVector{4}) = x
inner_coordinates(::KerrSchild, x::SVector{4}) = x
inner_coordinates(::River, x::SVector{4}) = x

function inner_coordinates(m::ConstantShift, x::SVector{4})
    t = x[1]
    β = m.shift
    return SVector{4}(t, x[2] + β[1] * t, x[3] + β[2] * t, x[4] + β[3] * t)
end

inner_coordinates(m::SpacetimeMetrics.TranslatedMetric, x::SVector{4}) =
    inner_coordinates(m.metric, x - m.distance)
inner_coordinates(m::SpacetimeMetrics.RotatedMetric, x::SVector{4}) =
    inner_coordinates(m.metric, m.R' * x)
inner_coordinates(m::SpacetimeMetrics.BoostedMetric, x::SVector{4}) =
    inner_coordinates(m.metric, m.Λ' * x)

function inner_coordinates(m::SpacetimeMetrics.GaugeWaveMetric, x::SVector{4})
    A, d = m.amplitude, m.period
    t, X, y, z = x
    φ = 2 * oftype(X, π) * (X - t) / d
    C = A * d / (4 * oftype(A, π))
    cφ = cos(φ)
    return inner_coordinates(m.metric, SVector(t - C * cφ, X + C * cφ, y, z))
end

function inner_coordinates(m::SpacetimeMetrics.SineShiftMetric, x::SVector{4})
    A, d = m.amplitude, m.period
    t, X, y, z = x
    φ = 2 * oftype(X, π) * (X - t) / d
    C = A * d / (2 * oftype(A, π))
    return inner_coordinates(m.metric, SVector(t, X + C * sin(φ), y, z))
end

function inner_coordinates(m::SpacetimeMetrics.ShiftedMinkowskiMetric, x::SVector{4})
    A, w = m.amplitude, m.width
    t, X, y, z = x
    return inner_coordinates(m.metric, SVector(t + A * w * tanh(X / w), X, y, z))
end

function inner_coordinates(m::SpacetimeMetrics.MovingGridMetric, x::SVector{4})
    V₀, w, xc = m.speed, m.width, m.center
    t, X, y, z = x
    V = V₀ * (1 - tanh((X - xc) / w)) / 2
    return inner_coordinates(m.metric, SVector(t, X + t * V, y, z))
end

"""
    base_chart(m::AbstractMetric) -> AbstractMetric

The innermost metric of a chain of SpacetimeMetrics wrappers — the chart
[`inner_coordinates`](@ref) maps into. [`ConstantShift`](@ref)'s is
`Minkowski()`.
"""
base_chart(m::AbstractMetric) = m
base_chart(::ConstantShift) = Minkowski()
base_chart(m::Union{SpacetimeMetrics.TranslatedMetric,SpacetimeMetrics.RotatedMetric,
                    SpacetimeMetrics.BoostedMetric,SpacetimeMetrics.GaugeWaveMetric,
                    SpacetimeMetrics.SineShiftMetric,
                    SpacetimeMetrics.ShiftedMinkowskiMetric,
                    SpacetimeMetrics.MovingGridMetric}) = base_chart(m.metric)

# --- the solutions --------------------------------------------------------------

"""
    ExactSolution

An exact solution of `□u = 0`, defined as a function `solution_value(sol,
x̂, Val(D))` of a point `x̂` of one base chart (`solution_chart(sol)`, a type)
in `D` spatial dimensions. Concrete solutions are `isbits`.
"""
abstract type ExactSolution end

"""
    PlaneWave(amplitude, k, phase = 0)

`u = A sin(k·x̂ − |k| t̂ + φ)` on Minkowski: a plane wave travelling along
the 3-vector `k`. At `D < 3` only the first `D` components of `k` should be
nonzero; the others multiply coordinates that are zero on the slice, and
would only change the frequency.
"""
struct PlaneWave{T} <: ExactSolution
    amplitude::T
    k::SVector{3,T}
    phase::T
end
function PlaneWave(A, k::AbstractVector, phase=0)
    T = float(promote_type(typeof(A), eltype(k), typeof(phase)))
    return PlaneWave{T}(T(A), SVector{3,T}(k), T(phase))
end
solution_chart(::PlaneWave) = Minkowski

@inline function solution_value(s::PlaneWave, x̂::SVector{4}, ::Val{D}) where {D}
    U = eltype(x̂)
    k = SVector{3,U}(s.k)
    ω = sqrt(dot(k, k))
    return U(s.amplitude) * sin(k[1] * x̂[2] + k[2] * x̂[3] + k[3] * x̂[4] - ω * x̂[1] +
                                U(s.phase))
end

"""
    PlanePulse(amplitude, n, width, offset = 0)

`u = A exp(−((n·x̂ − t̂ − s)/w)²)` on Minkowski, `n` a unit 3-vector: a
Gaussian pulse travelling along `n` at the speed of light, centered on the
plane `n·x̂ = s` at `t̂ = 0`. A localized feature that moves, for the
regrid tests. As for [`PlaneWave`](@ref), at `D < 3` the components of `n`
along dropped axes should be zero.
"""
struct PlanePulse{T} <: ExactSolution
    amplitude::T
    n::SVector{3,T}
    width::T
    offset::T
end
function PlanePulse(A, n::AbstractVector, w, s=0)
    T = float(promote_type(typeof(A), eltype(n), typeof(w), typeof(s)))
    nn = SVector{3,T}(n)
    nn = nn / sqrt(dot(nn, nn))
    return PlanePulse{T}(T(A), nn, T(w), T(s))
end
solution_chart(::PlanePulse) = Minkowski

@inline function solution_value(s::PlanePulse, x̂::SVector{4}, ::Val{D}) where {D}
    U = eltype(x̂)
    n = SVector{3,U}(s.n)
    # Normalized here as well as in the constructor: retyped to another
    # precision, the stored `n` is a unit vector only to that precision, and
    # the pulse must still travel at exactly the speed of light.
    nn = sqrt(dot(n, n))
    ξ = ((n[1] * x̂[2] + n[2] * x̂[3] + n[3] * x̂[4]) / nn - x̂[1] - U(s.offset)) / U(s.width)
    return U(s.amplitude) * exp(-ξ * ξ)
end

"""
    PolynomialWave(amplitude, scale)

`u = A (t̂ x̂ + (t̂² + x̂²)/2) / L²` on Minkowski, with `L` the `scale`: a
quadratic solution in every dimension (`□u = −1 + 1`). Every stencil here is
exact on it, so the right-hand side at the exact state is exact to rounding.
A run's error is not: RK4's stage order is one, and the Dirichlet data
depend on time. It needs no transcendental function, so it is the solution
of the software-float type tests, where MultiFloats has no `sin`.
"""
struct PolynomialWave{T} <: ExactSolution
    amplitude::T
    scale::T
end
function PolynomialWave(A, L)
    T = float(promote_type(typeof(A), typeof(L)))
    return PolynomialWave{T}(T(A), T(L))
end
solution_chart(::PolynomialWave) = Minkowski

@inline function solution_value(s::PolynomialWave, x̂::SVector{4}, ::Val{D}) where {D}
    U = eltype(x̂)
    t, x = x̂[1], x̂[2]
    L = U(s.scale)
    return U(s.amplitude) * (t * x + (t * t + x * x) / 2) / (L * L)
end

"""
    StaticHole(M, a)

The static monopole on Kerr in Kerr–Schild coordinates,
`u = ln((r − r₊)/(r − r₋))` with `r₊,₋ = M ± √(M² − a²)` and `r` the
spheroidal radius. It is a function of Boyer–Lindquist `r` alone, which is
Kerr–Schild's `r`, and solves `∂_r(Δ ∂_r u) = 0` with `Δ = (r − r₊)(r − r₋)`;
Kerr–Schild's `t` and `φ` differ from Boyer–Lindquist's by functions of `r`,
which a stationary, axisymmetric `u` does not see. At `a = 0` it is
`ln(1 − 2M/r)`. Singular on the horizon, so the domain it is evolved on
must stay outside `r₊` (`CODE.md`, "Exact solutions"). Three dimensions
only.

Wrapped in `boost` or `translate`, it is an exact solution on a moving hole.
"""
struct StaticHole{T} <: ExactSolution
    mass::T
    spin::T
end
function StaticHole(M, a=0)
    T = float(promote_type(typeof(M), typeof(a)))
    return StaticHole{T}(T(M), T(a))
end
solution_chart(::StaticHole) = KerrSchild

"""
    kerr_schild_radius(a, x̂) -> r

The Kerr–Schild spheroidal radius at the point `x̂ = (t, x, y, z)`:
`r⁴ − r²(ρ² − a²) − a²z² = 0`, `ρ² = x² + y² + z²` — the expression
SpacetimeMetrics' `KerrSchild` uses.
"""
@inline function kerr_schild_radius(a, x̂::SVector{4})
    x, y, z = x̂[2], x̂[3], x̂[4]
    ρ2 = x^2 + y^2 + z^2
    return sqrt((ρ2 - a^2 + sqrt(4 * a^2 * z^2 + (ρ2 - a^2)^2)) / 2)
end

@inline function solution_value(s::StaticHole, x̂::SVector{4}, ::Val{D}) where {D}
    U = eltype(x̂)
    M, a = U(s.mass), U(s.spin)
    r = kerr_schild_radius(a, x̂)
    δ = sqrt(M^2 - a^2)
    return log((r - (M + δ)) / (r - (M - δ)))
end

"""
    StaticRiver(M)

The static monopole on the [`River`](@ref) metric in `D` dimensions: the
solution of `∂_r(r^{D−1}(1 − 2M/r) ∂_r u) = 0`, with `r` the radius in the
first `D` coordinates and the integration constant fixed by `∂_r u =
1/(r^{D−1}(1 − 2M/r))`:

| `D` | `u` |
|---|---|
| 1 | `r + 2M ln((r − 2M)/M)` |
| 2 | `ln((r − 2M)/M)` |
| 3 | `ln(1 − 2M/r)/(2M)` |

At `D = 3` this is [`StaticHole`](@ref)'s `a = 0` monopole (scaled by
`1/2M`) in Painlevé–Gullstrand coordinates. Singular on the horizon `r = 2M`.
"""
struct StaticRiver{T} <: ExactSolution
    mass::T
end
StaticRiver(M::Integer) = StaticRiver(float(M))
solution_chart(::StaticRiver) = River

@inline function solution_value(s::StaticRiver, x̂::SVector{4}, ::Val{D}) where {D}
    U = eltype(x̂)
    M = U(s.mass)
    r = sqrt(x̂[2]^2 + x̂[3]^2 + x̂[4]^2)
    if D == 1
        return r + 2M * log((r - 2M) / M)
    elseif D == 2
        return log((r - 2M) / M)
    else
        return log(1 - 2M / r) / (2M)
    end
end

"""
    check_solution(sol::ExactSolution, bg::Background)

Throw an `ArgumentError` unless `sol` is defined in `bg`'s base chart (and,
for the Kerr monopole, in three dimensions). Host-side, once per case.
"""
function check_solution(sol::ExactSolution, bg::Background{D}) where {D}
    chart = base_chart(bg.metric)
    chart isa solution_chart(sol) || throw(ArgumentError(
        "the solution $(nameof(typeof(sol))) is defined in the chart of " *
        "$(nameof(solution_chart(sol))), but this background's base chart is " *
        "$(nameof(typeof(chart))): an exact solution transfers through a chart " *
        "map only, not between spacetimes"))
    sol isa StaticHole && D != 3 && throw(ArgumentError(
        "the Kerr monopole is a solution in three dimensions; at D = $D the slice " *
        "of Kerr–Schild is a model spacetime, not Kerr"))
    # To rounding: a background written in rationals and a solution in
    # floats, or either retyped, name the same hole.
    same(a, b) = isapprox(a, b; rtol=8 * eps(float(typeof(a))), atol=0)
    if sol isa StaticHole && chart isa KerrSchild
        (same(sol.mass, chart.mass) && same(sol.spin, chart.spin)) || throw(ArgumentError(
            "the monopole's (M, a) = ($(sol.mass), $(sol.spin)) differ from the " *
            "background's ($(chart.mass), $(chart.spin))"))
    end
    if sol isa StaticRiver && chart isa River
        same(sol.mass, chart.mass) || throw(ArgumentError(
            "the monopole's M = $(sol.mass) differs from the river's $(chart.mass)"))
    end
    return nothing
end

# The tag of the exact state's derivative pass.
struct ExactTag end

"""
    exact_value(sol, bg::Background{D}, t, x) -> u

The solution's value at time `t` and position `x` of `bg`'s chart.
"""
@inline function exact_value(sol::ExactSolution, bg::Background{D}, t, x) where {D}
    xs = SVector{D}(x)
    p = spacetime_point(Val(D), convert(eltype(xs), t), xs)
    return solution_value(sol, inner_coordinates(bg.metric, p), Val(D))
end

"""
    exact_state(sol, bg::Background{D}, t, x) -> (u, Π)

The state of the exact solution at time `t` and position `x` of `bg`'s
chart, with `Π = √γ n^a ∂_a u = (√γ/α)(∂_t u − β^i ∂_i u)`
(`CODE.md`, "The equations"). The derivatives of `u` come from one
forward-mode pass with `D + 1` partials through the chart map and the
solution. `isbits` and allocation-free, so it runs in the Dirichlet
boundary kernel and in `fill_by_coordinates!`.
"""
@inline function exact_state(sol::ExactSolution, bg::Background{D}, t, x) where {D}
    xs = SVector{D}(x)
    T = eltype(xs)
    z = seed(ExactTag, SVector{D + 1}(convert(T, t), xs...))
    p = spacetime_point(Val(D), z[1], SVector{D}(ntuple(i -> z[i + 1], Val(D))))
    ud = solution_value(sol, inner_coordinates(bg.metric, p), Val(D))
    u = ForwardDiff.value(ud)
    ∂u = ForwardDiff.partials(ud)
    αsγ, β, _ = wave_fields(spacetime_metric(bg, convert(T, t), xs))
    s = ∂u[1]
    for i in 1:D
        s -= β[i] * ∂u[i + 1]
    end
    return (u, s / αsγ)
end

"""
    wave_residual(sol, bg::Background{D}, t, x) -> (□u, scale)

`∂_μ(√−g g^{μν} ∂_ν u)` for the exact solution at time `t` and position `x`,
by nested forward-mode differentiation of the flux form — independent of the
coefficients and of [`exact_state`](@ref) — and the largest component of the
flux, the scale `□u` is small against. Host-side; it allocates.
"""
function wave_residual(sol::ExactSolution, bg::Background{D}, t, x) where {D}
    z = SVector{D + 1}(t, x...)
    u(y) = exact_value(sol, bg, y[1], SVector{D}(ntuple(i -> y[i + 1], D)))
    function F(y)
        g = spacetime_metric(bg, y[1], SVector{D}(ntuple(i -> y[i + 1], D)))
        return sqrt(-det(g)) * (inv(g) * ForwardDiff.gradient(u, y))
    end
    return tr(ForwardDiff.jacobian(F, z)), maximum(abs, F(z))
end
