# The finite-difference and Kreiss–Oliger weights, built in exact rational
# arithmetic and rounded once into `T`, and the kernel-side contractions that
# apply them to the working array.
#
# Ported from TreeGeneralizedHarmonic's `src/stencils.jl` (the weights) and
# its `src/evolution.jl` (`axis_stencil`, `mixed_stencil`, `work_strides`),
# with the contractions generalized from three dimensions to `D`. The weights
# are the same numbers, so a stencil here and a stencil there are the same
# operator — which is what lets a closure developed on this package's scalar
# wave carry over to the generalized-harmonic system (`CODE.md`, "Extension
# points").
#
# What the rest of the package relies on (`CODE.md`, "Discretization"):
#
#   * **The weights are for unit spacing.** `derivative_weights` returns the
#     weights of `∂^m` on samples one apart; the physical operator divides by
#     `h^m`. `dissipation_weights` returns the weights of
#     `(−1)^{r+1} 2^{−2r} (Δ_+Δ_−)^r`, so the physical operator is
#     `(ε/h)` times the contraction.
#   * **They are `isbits` and free at run time.** Both are `@generated`: the
#     `Rational{BigInt}` construction happens once per `(q, m)` at compile
#     time, and the emitted code holds each weight's exact numerator and
#     denominator as `Int` literals with one division between them.
#   * **The conversion into `T` happens at the call site, not in the
#     generator.** A generator may only call methods that existed when it was
#     defined, and this package is precompiled before a driver loads
#     MultiFloats; converting in the generator throws "method too new" at
#     `Float32x2` (measured in TreeGeneralizedHarmonic). Emitting
#     `T(num)/T(den)` moves the conversion into the caller's world.
#   * **`q` is even.** Everything is centered, and the ghost width
#     `G = q/2 + 1` is one more than the derivatives need, because the
#     dissipation reaches one point further.

# `BigInt` for the same reason TreeAMR uses it: `Rational` arithmetic is
# checked, so an overflowing intermediate would be a hard error rather than
# a wrong answer, and a bignum removes the ceiling instead of moving it.
# This runs at compile time, never per point.
const StencilRational = Rational{BigInt}

# Multiply the polynomial `c` (coefficients low to high) by `(x + a)`.
function _polymul_linear(c::Vector{StencilRational}, a::StencilRational)
    out = Vector{StencilRational}(undef, length(c) + 1)
    out[1] = a * c[1]
    for i in 2:length(c)
        out[i] = c[i - 1] + a * c[i]
    end
    out[end] = c[end]
    return out
end

"""
    lagrange_derivative_weights(nodes, m) -> Vector{Rational{BigInt}}

Weights `w` with `sum(w[i] * u(nodes[i])) == u^(m)(0)` for every polynomial
`u` of degree less than `length(nodes)`.

The `m`-th derivative at the origin of the Lagrange basis on `nodes`,
computed by building each basis polynomial's numerator
`∏_{k≠i} (x − x_k)` coefficient by coefficient and reading off `m! · c_m`.
Nodes are exact rationals and so is the result; that exactness is the whole
point, and it is why this is written out here rather than taken from a
Vandermonde solve.

Exactness holds for *any* distinct nodes, so this is also what a one-sided
or shifted stencil would be built from if one were ever wanted. The
centered stencils this package uses are [`derivative_weights`](@ref).
"""
function lagrange_derivative_weights(nodes::AbstractVector{<:Rational},
                                     m::Integer)
    n = length(nodes)
    m >= 0 || throw(ArgumentError("a derivative order must be non-negative, " *
                                  "but m=$m"))
    ns = StencilRational.(nodes)
    ws = Vector{StencilRational}(undef, n)
    for i in 1:n
        c = [one(StencilRational)]
        den = one(StencilRational)
        for k in 1:n
            k == i && continue
            c = _polymul_linear(c, -ns[k])
            den *= (ns[i] - ns[k])
        end
        # `u^(m)(0) = m! · c_m`, and a polynomial of degree below `m` has no
        # such coefficient at all.
        ws[i] = m < n ? factorial(big(m)) * c[m + 1] / den :
                zero(StencilRational)
    end
    return ws
end

"""
    rational_derivative_weights(q, m) -> Vector{Rational{BigInt}}

The exact weights behind [`derivative_weights`](@ref): the centered `q`-th
order stencil for `∂^m` on the `q + 1` integer nodes `−q/2 … q/2`.

Exposed because exactness is a claim a test should be able to make in
`Rational` rather than through a tolerance — the textbook tables, the
polynomial degrees each operator is and is not exact on, and the
bit-identity of the conversion into `Float64`, all of which
`test/stencils_tests.jl` asserts here rather than on the rounded weights.
"""
function rational_derivative_weights(q::Integer, m::Integer)
    q >= 2 && iseven(q) || throw(ArgumentError(
        "a centered stencil needs an even order q ≥ 2 so that its half-width " *
        "q/2 is an integer and CODE.md's ghost width G = q/2 + 1 covers it, " *
        "but q=$q"))
    m in (1, 2) || throw(ArgumentError(
        "this package differentiates at most twice — CODE.md's second-order " *
        "reduction takes ∂_i and ∂_i∂_j and nothing else — but m=$m"))
    r = q ÷ 2
    return lagrange_derivative_weights([StencilRational(j) for j in (-r):r], m)
end

"""
    rational_dissipation_weights(r) -> Vector{Rational{BigInt}}

The exact weights behind [`dissipation_weights`](@ref): the `2r + 1` entries
of `(−1)^{r+1} 2^{−2r} (Δ_+Δ_−)^r`, the undivided Kreiss–Oliger operator on
unit-spaced samples.

`(Δ_+Δ_−)^r` is the `2r`-th undivided central difference, whose coefficient
at offset `j` is `(−1)^{r−j} binom(2r, r−j)`; the alternating sign
`(−1)^{r+1}` of `CODE.md`'s formula makes the center coefficient negative at
every `r`, which is what makes the operator damping rather than driving.
"""
function rational_dissipation_weights(r::Integer)
    r >= 1 || throw(ArgumentError(
        "the Kreiss–Oliger operator of order 2r = q + 2 needs a rank r ≥ 1 " *
        "to have a stencil at all — and r ≥ 2 at every order q ≥ 2 this " *
        "package uses — but r=$r"))
    sgn = iseven(r + 1) ? 1 : -1
    den = StencilRational(big(2)^(2r))
    return [sgn * (-1)^(r - j) * StencilRational(binomial(2r, r - j)) / den
            for j in (-r):r]
end

# The body every weight method returns: one `T(num)/T(den)` per entry, with
# the exact numerator and denominator spliced as `Int` literals.
#
# The division is the *only* rounding in the construction, and it happens in
# the caller's world at the caller's type. For an IEEE type it is correctly
# rounded — bit-identical to converting the exact rational, since `num` and
# `den` are small integers and therefore exact in `T` — and for a software
# type it is correct to that type's own division accuracy, which is one ulp
# of something far below anything this package measures. Doing it the other
# way, converting in the generator, is what the header comment records as
# failing at `Float32x2`.
function _weight_expr(ws::Vector{StencilRational})
    entries = [:(T($(Int(numerator(w)))) / T($(Int(denominator(w)))))
               for w in ws]
    return :(SVector{$(length(ws)),T}($(entries...)))
end

"""
    derivative_weights([T = Float64], ::Val{q}, ::Val{m}) -> SVector{q+1,T}

The centered finite-difference weights of order `q` for the `m`-th
derivative, `m ∈ {1, 2}`, on `q + 1` samples **one apart**, in the offset
order `−q/2 … q/2`.

The physical operator divides by the spacing: `∂^m u ≈ apply_stencil(w, u,
i) / h^m`, with `h` the block's own spacing. Keeping `h` out of the weights
is what lets one weight vector serve every refinement level, and it is the
form the streaming kernel wants — the `1/h^m` are per-block coefficients
read once per point, the weights are compile-time constants.

`q` is even, and `Val`-wrapped because the kernel specialises on it
(the `Val`s are built once per problem, never per evaluation).
The first derivative's center weight is exactly zero and is returned anyway,
so that both operators have the same layout; a kernel may skip it.

Order, in the sense a convergence test measures: the first derivative is
exact on polynomials of degree `≤ q` and not `q + 1`; the second, on degree
`≤ q + 1` — one better than it was built for, by the symmetry of an even
`q` — and not `q + 2`. Both therefore carry a truncation error `O(h^q)`.

The **mixed** derivative `∂_i∂_j`, `i ≠ j`, has no weights of its own: it is
the tensor product of two of these first-derivative vectors, one per axis,
which is the form that reads the edge and corner ghosts TreeAMR fills
unconditionally (`CODE.md`, "Discretization"). The reference
contraction is [`apply_mixed_stencil`](@ref), and its docstring says which
end the sum is formed at.

Built in exact rational arithmetic and rounded once into `T`; see
[`rational_derivative_weights`](@ref). The method is `@generated`, so the
rationals exist only while it compiles: what it emits is each weight's exact
numerator and denominator as `Int` literals with one division between them,
and the result is an `isbits` `SVector` a device kernel holds in registers.
At `Float64` and `Float32` that division is correctly rounded — the same
bits as converting the exact rational — and folds to a constant before the
kernel runs.
"""
@generated function derivative_weights(::Type{T}, ::Val{q},
                                       ::Val{m}) where {T,q,m}
    ws = try
        rational_derivative_weights(q, m)
    catch err
        err isa ArgumentError || rethrow()
        return :(throw($err))
    end
    return _weight_expr(ws)
end

@inline derivative_weights(q::Val, m::Val) = derivative_weights(Float64, q, m)

"""
    dissipation_weights([T = Float64], ::Val{r}) -> SVector{2r+1,T}

The Kreiss–Oliger dissipation weights of order `2r = q + 2` on `2r + 1`
samples **one apart**, in the offset order `−r … r`.

`CODE.md`, "Discretization", fixes the operator as

    Q_d u = ε (−1)^{r+1} (h_d^{2r−1} / 2^{2r}) (D_+ D_−)^r u,   r = q/2 + 1

with `h_d` the block's own spacing. What is in the weights and what is not:

| factor | where it is applied |
|---|---|
| `(−1)^{r+1}` | **in the weights** |
| `2^{−2r}` | **in the weights** |
| `(D_+D_−)^r`, the undivided part | **in the weights** |
| `h_d^{2r−1}` against `(D_+D_−)^r`'s own `h_d^{−2r}` | by the caller, as a single `1/h_d` |
| `ε` | by the caller |

so that the whole operator is `Q_d u = (ε / h_d) * apply_stencil(w, u, i)`.
One factor of the spacing, not `2r − 1` of them: that is the point of the
`h_d^{2r−1}` in the formula, and it is why a refinement level's dissipation
scales with its own resolution and `ε ∈ (0, 1)` is neutral to the CFL
condition.

**The sign is damping**, and it is carried by the weights: the center weight
is `−binom(2r, r)/2^{2r} < 0` at every `r`, and on a Fourier mode
`u_j = exp(i k x_j)` the contraction is

    apply_stencil(w, u, i) = −sin^{2r}(k h_d / 2) · u_i    ≤ 0 · u_i

so `Q_d u = −(ε/h_d) sin^{2r}(k h_d/2) u`. The Nyquist mode `k h_d = π` is
damped at exactly `ε/h_d` and the smooth modes are barely touched, which is
the normalisation the formula's `2^{−2r}` exists for. `Q_d` is *added* to
the right-hand side; flipping the sign turns the term into an amplifier of
exactly the grid-scale noise it is there to remove, and the symptom is a run
that blows up faster with larger `ε`.

`Q_d` annihilates polynomials of degree `< 2r`, so it contributes
`O(h^{2r−1}) = O(h^{q+1})` to the truncation error and does not degrade a
`q`-th order scheme. Its stencil reaches `r = q/2 + 1` points, one further
than the derivatives, which is where `CODE.md`'s ghost width `G = q/2 + 1`
comes from.

Wherever `g^{dd} < 0` (a shift superluminal along a grid line, as inside a
horizon) the second-order-in-space scheme needs dissipation to be stable
(Calabrese 2004; TreeExcision's `notes/literature.md`), so `ε` is a case
parameter: a smooth run on a subluminal background needs little or none.

Built in exact rational arithmetic and rounded once into `T`; see
[`rational_dissipation_weights`](@ref) and [`derivative_weights`](@ref) for
what the `@generated` method emits. Every entry here is dyadic, so the
conversion is exact in any binary floating-point type.
"""
@generated function dissipation_weights(::Type{T}, ::Val{r}) where {T,r}
    ws = try
        rational_dissipation_weights(r)
    catch err
        err isa ArgumentError || rethrow()
        return :(throw($err))
    end
    return _weight_expr(ws)
end

@inline dissipation_weights(r::Val) = dissipation_weights(Float64, r)

"""
    dissipation_rank(::Val{q}) -> Val{r}

The rank `r = q/2 + 1` of the dissipation operator that goes with a `q`-th
order scheme, so that no call site has to spell `CODE.md`'s relation
`2r = q + 2` again.

The same number is the ghost width `G`, and for the same reason: the
dissipation's stencil is the widest one an evaluation takes, reaching `r`
points where the derivatives reach `q/2`.
"""
@inline dissipation_rank(::Val{q}) where {q} = Val(q ÷ 2 + 1)

"""
    apply_stencil(w, u::AbstractVector, i) -> scalar
    apply_stencil(w, f, x, h) -> scalar

The centered contraction of a weight vector: `sum(w[k] * u[i + k − 1 − r])`
over the `2r + 1` entries of `w`, either against samples stored in `u` around
index `i`, or against a callable `f` sampled at `x + j·h` for `j = −r … r`.

Host-side, and for the tests: this is the reference the weights are measured
with, and the definition [`axis_stencil`](@ref) has to agree with
while never materialising `u` or `w` as anything but registers. Nothing in
an evaluation calls it.

It returns the raw contraction. The scale factor is the caller's, and which
one it is depends on the weights: `/h^m` for
[`derivative_weights`](@ref), `ε/h` for [`dissipation_weights`](@ref).

Summation runs from the lowest offset to the highest, which is a choice and
not a law — a different order differs in the last place, and `CODE.md`'s
"Measured results" records what that does and does not mean for the
bit-identity the threading test asserts.
"""
@inline function apply_stencil(w::SVector{n}, u::AbstractVector,
                               i::Integer) where {n}
    r = (n - 1) ÷ 2
    s = w[1] * u[i - r]
    for k in 2:n
        s += w[k] * u[i - r + k - 1]
    end
    return s
end

@inline function apply_stencil(w::SVector{n}, f, x, h) where {n}
    r = (n - 1) ÷ 2
    s = w[1] * f(x - r * h)
    for k in 2:n
        s += w[k] * f(x + (k - 1 - r) * h)
    end
    return s
end

"""
    apply_mixed_stencil(w, f, x, y, hx, hy) -> scalar

The mixed derivative as the tensor product of two first-derivative weight
vectors: `sum_a w[a] * sum_b w[b] * f(x + a·hx, y + b·hy)`, the reference
for `∂_i∂_j` with `i ≠ j`.

`CODE.md`, "Discretization": the mixed derivative has no weights
of its own. The same vector serves both axes, and the `(q+1)²` product is
never built — it is a sum of sums, which is what lets a kernel form it one
line at a time.

**Which end it is formed at**: the *inner* sum runs along the second axis
`y`, the *outer* along the first axis `x`. The two orders are equal in exact
arithmetic and differ in the last place in floating point, so the order is
part of the operator, not an implementation detail; [`mixed_stencil`](@ref) picks
one and keeps it.

As with [`apply_stencil`](@ref) the result is the raw contraction: the
physical `∂_x∂_y f` divides it by `hx·hy`.
"""
@inline function apply_mixed_stencil(w::SVector, f, x, y, hx, hy)
    return apply_stencil(w, ξ -> apply_stencil(w, η -> f(ξ, η), y, hy), x, hx)
end

# --- the kernel-side stencil contractions -----------------------------------
#
# `apply_stencil` and `apply_mixed_stencil` above are the host-side reference
# *definitions*; these are what the kernel evaluates, and the two agree bit for
# bit (`test/stencils_tests.jl`). Both sum from the lowest offset to the
# highest, and the mixed one runs its inner sum along the second axis.
#
# They address the working array by a **linear index**: a base index for the
# point and one stride per axis, rather than a `(i, j, k, v, b)` tuple per load.
# The working array is dense and column-major (TreeAMR allocates it), so the
# two name the same element. TreeGeneralizedHarmonic measured the linear form
# 2.4x faster than the cartesian one for bit-identical output.
#
# Everything here is `@generated` or written without closures: a device
# compiles an `ntuple(Val(n)) do … end` closure as a real call unless inlining
# is forced, and a generated method is inlined only if its body carries an
# explicit `:inline` meta.

"""
    axis_stencil(w, work, base, stride) -> scalar

`∑_k w[k] work[base + (k − 1 − r)·stride]` with `r = (n − 1) ÷ 2`: the
contraction every one-dimensional operator here is, written out term by term
from the lowest offset to the highest. The raw contraction on unit spacing,
as [`apply_stencil`](@ref) is.
"""
@generated function axis_stencil(w::SVector{n,T}, work, base::Int,
                                 stride::Int) where {n,T}
    r = (n - 1) ÷ 2
    ex = :(w[1] * work[base + $(-r) * stride])
    for k in 2:n
        ex = :($ex + w[$k] * work[base + $(k - 1 - r) * stride])
    end
    return Expr(:block, Expr(:meta, :inline), :(@inbounds $ex))
end

"""
    mixed_stencil(w, work, base, s1, s2) -> scalar

The mixed derivative as the tensor product of two first-derivative vectors:
outer sum along the axis with stride `s1`, inner along `s2`, as
[`apply_mixed_stencil`](@ref) is. It reads the edge ghosts TreeAMR fills
unconditionally, which is why `∂_i∂_j` needs no wider halo than `∂_i∂_i`.
"""
@generated function mixed_stencil(w::SVector{n,T}, work, base::Int, s1::Int,
                                  s2::Int) where {n,T}
    r = (n - 1) ÷ 2
    outer = :(nothing)
    for a in 1:n
        inner = :(w[1] * work[base + $(a - 1 - r) * s1 + $(-r) * s2])
        for e in 2:n
            inner = :($inner + w[$e] * work[base + $(a - 1 - r) * s1 + $(e - 1 - r) * s2])
        end
        outer = a == 1 ? :(w[1] * $inner) : :($outer + w[$a] * $inner)
    end
    return Expr(:block, Expr(:meta, :inline), :(@inbounds $outer))
end

"""
    work_strides(work, ::Val{D}) -> (st::NTuple{D,Int}, sv, sb)

The strides of a dense array laid out as TreeAMR's working array,
`(n_1, …, n_D, nvars, nblocks)`: one per axis, then the stride between
variables and between blocks. From `size` alone, so it holds for any dense
array on any backend, and for the state layout `(N, …, N, nvars, nblocks)` of
`du` as well.
"""
@generated function work_strides(work, ::Val{D}) where {D}
    st = Any[1]
    for d in 2:D
        push!(st, :($(st[end]) * size(work, $(d - 1))))
    end
    sv = :($(st[end]) * size(work, $D))
    return quote
        $(Expr(:meta, :inline))
        sv = $sv
        return $(Expr(:tuple, st...)), sv, sv * size(work, $(D + 1))
    end
end

"""
    linear_index(I, G, st, sb, ::Val{D}) -> Int

The linear index of the first variable at point `I = (i_1, …, i_D, b)` of an
array with strides `st` and block stride `sb`, after shifting each `i_d` by
`G[d]` — the owned point's stored index for the working array, or `G = 0` for
a state-layout array. Add `(v − 1)·sv` for variable `v`.
"""
@generated function linear_index(I, G, st, sb::Int, ::Val{D}) where {D}
    ex = :(1 + (I[$(D + 1)] - 1) * sb)
    for d in 1:D
        ex = :($ex + (I[$d] + G[$d] - 1) * st[$d])
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end
