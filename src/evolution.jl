# The right-hand side (`CODE.md`, "The right-hand side").
#
#     ∂_t u = β^i ∂_i u + (α/√γ) Π
#     ∂_t Π = β^i ∂_i Π + (∂_i β^i) Π + A^{ij} ∂_i ∂_j u + (∂_i A^{ij}) ∂_j u
#
# plus Kreiss–Oliger dissipation on both. One kernel over every owned point,
# `wave_rhs_kernel!`, which forms the point's position, takes the coefficients
# from the coefficient source, and calls `wave_rhs_point` — the pointwise
# operator, generic over a `StencilProvider`, that TreeExcision's own kernels
# call with their closures.

"""
    wave_rhs_point(S::StencilProvider, c::WaveCoefficients, work, bu, bΠ, inv_h,
                   εh, ::Val{D}, ::Val{DISS}) -> (∂ₜu, ∂ₜΠ)

The right-hand side at one point: `bu` and `bΠ` are the linear indices of `u`
and `Π` at the point in the working array `work`, `inv_h` the block's inverse
spacing, `εh = ε/h` the dissipation's scale (read only when `DISS`). The
stencils come from `S`; the main kernel passes [`Centered`](@ref), and an
excision kernel passes a provider whose stencils stop at the excision
surface.

The sums run in a fixed order — advection, the divergence terms and the
diagonal second derivatives axis by axis, then the mixed derivatives `i < j`
row by row, then the dissipation — which is part of the operator: a
different order differs in the last place.
"""
@inline function wave_rhs_point(S::StencilProvider, c::WaveCoefficients, work,
                                bu::Int, bΠ::Int, inv_h, εh, ::Val{D},
                                ::Val{DISS}) where {D,DISS}
    Π = @inbounds work[bΠ]
    inv_h2 = inv_h * inv_h
    ∂ₜu = c.αsγ * Π
    ∂ₜΠ = c.divβ * Π
    for d in 1:D
        βd = c.β[d]
        ∂u = d1(S, work, bu, d) * inv_h
        ∂Π = d1(S, work, bΠ, d) * inv_h
        ∂ₜu += βd * adv(S, βd, ∂u, work, bu, d)
        ∂ₜΠ += βd * adv(S, βd, ∂Π, work, bΠ, d)
        ∂ₜΠ += c.divA[d] * ∂u
        ∂ₜΠ += c.A[d, d] * (d2(S, work, bu, d) * inv_h2)
    end
    for i in 1:D, j in (i + 1):D
        ∂ₜΠ += 2 * c.A[i, j] * (dmix(S, work, bu, i, j) * inv_h2)
    end
    if DISS
        for d in 1:D
            ∂ₜu += εh * ko(S, work, bu, d)
            ∂ₜΠ += εh * ko(S, work, bΠ, d)
        end
    end
    return ∂ₜu, ∂ₜΠ
end

# The position of an owned point, formed exactly as TreeAMR's
# `fill_by_coordinates!` kernel forms it — the same origin, spacing and
# expression in the same order — so that the coefficients the analytic source
# evaluates here are at the positions the sampled set was filled at. Every
# field set here is vertex-centered, where TreeAMR's offset is `one(h)`.
@inline function point_position(origin, h, I, ::Val{D}) where {D}
    off = one(h)
    return ntuple(d -> origin[d] + (I[d] - off) * h, Val(D))
end

@kernel function wave_rhs_kernel!(du, @Const(work), cwork, bg, @Const(origins),
                                  @Const(spacings), t, ε, ::Val{D}, ::Val{G},
                                  ::Val{q}, ::Val{DISS}) where {D,G,q,DISS}
    I = @index(Global, NTuple)
    b = I[D + 1]
    T = eltype(du)
    h = spacings[b]
    x = point_position(origins[b], h, I, Val(D))
    st, sv, sb = work_strides(work, Val(D))
    bu = linear_index(I, G, st, sb, Val(D))
    c = point_coefficients(cwork, bg, I, x, t, Val(D))
    inv_h = inv(h)
    εh = DISS ? ε * inv_h : zero(T)
    ∂ₜu, ∂ₜΠ = wave_rhs_point(Centered(T, Val(q), st), c, work, bu, bu + sv, inv_h,
                              εh, Val(D), Val(DISS))
    sst, ssv, ssb = work_strides(du, Val(D))
    o = linear_index(I, ntuple(_ -> 0, Val(D)), sst, ssb, Val(D))
    @inbounds du[o] = ∂ₜu
    @inbounds du[o + ssv] = ∂ₜΠ
end

"""
    is_stationary(m::AbstractMetric) -> Bool

Whether the metric's components do not depend on `t` in its chart, so that
its coefficients may be sampled once per mesh. `false` for anything not
known to be stationary — the analytic source is always correct.
"""
is_stationary(::AbstractMetric) = false
is_stationary(::Union{Minkowski,KerrSchild,River,ConstantShift}) = true
is_stationary(m::SpacetimeMetrics.ShiftedMinkowskiMetric) = is_stationary(m.metric)
is_stationary(m::SpacetimeMetrics.RotatedMetric) = is_stationary(m.metric)
is_stationary(m::SpacetimeMetrics.TranslatedMetric) = is_stationary(m.metric)
is_stationary(m::SpacetimeMetrics.BoostedMetric) =
    iszero(m.velocity) && is_stationary(m.metric)

"""
    WaveProblem(fs, schedule, case; q = 4, coefficients = :auto)

What the integrator carries: the state field set `fs` (two variables, `u` and
`Π`, vertex-centered, `G ≥ q/2 + 1`), its ghost schedule, the
[`WaveCase`](@ref), and everything invariant hoisted out of the
per-evaluation path — the per-block geometry on the field set's backend, the
coefficient source, and the kernel's switches as `Val`s.

`coefficients` chooses the source (`CODE.md`, "Coefficient sources"):
`:sampled` fills a coefficient field set once, here, and is refused for a
background that is not [`is_stationary`](@ref); `:analytic` evaluates the
coefficients in the kernel; `:auto` takes `:sampled` where it is allowed.
A problem belongs to one mesh: after a regrid, build a new one.
"""
struct WaveProblem{T,D,G,q,DISS,F,S,C,CW,O,P}
    fs::F
    schedule::S
    case::C
    cfs::CW                     # the coefficient field set, or `nothing`
    origins::O
    spacings::P
    ε::T
end

function WaveProblem(fs::FieldSet{T,D}, schedule::GhostSchedule, case;
                     q::Integer=4, coefficients::Symbol=:auto) where {T,D}
    fs.nvars == 2 || throw(ArgumentError(
        "the state has two variables, u and Π, but this field set has $(fs.nvars)"))
    fs.centering == vertexcentered(D) || throw(ArgumentError(
        "the state is vertex-centered, which `point_position` assumes, but this " *
        "field set is $(fs.centering)"))
    q >= 2 && iseven(q) || throw(ArgumentError("q must be even and at least 2, got $q"))
    case = retype(T, case)
    all(>=(q ÷ 2 + 1), fs.G) || throw(ArgumentError(
        "order q = $q needs G ≥ q/2 + 1 = $(q ÷ 2 + 1) — the dissipation reaches " *
        "one point past the derivatives — but G = $(fs.G)"))
    ndims(case.background) == D || throw(ArgumentError(
        "the case's background is seen in $(ndims(case.background)) dimensions, " *
        "the mesh has $D"))
    coefficients in (:auto, :sampled, :analytic) || throw(ArgumentError(
        "coefficients must be :auto, :sampled or :analytic, got :$coefficients"))
    stationary = is_stationary(case.background.metric)
    coefficients === :sampled && !stationary && throw(ArgumentError(
        "sampled coefficients are evaluated once per mesh, but this background " *
        "($(nameof(case.background.metric))) is not known to be stationary"))
    backend = get_backend(fs.work)
    cfs = if coefficients === :sampled || (coefficients === :auto && stationary)
        fill_coefficients!(coefficient_fieldset(fs.forest; backend=backend),
                           case.background, zero(T))
    else
        nothing
    end
    origins = to_backend(backend, block_origins(fs.forest, T))
    spacings = to_backend(backend, block_spacings(fs.forest, T))
    DISS = !iszero(case.ε)
    return WaveProblem{T,D,fs.G,Int(q),DISS,typeof(fs),typeof(schedule),typeof(case),
                       typeof(cfs),typeof(origins),typeof(spacings)}(
        fs, schedule, case, cfs, origins, spacings, convert(T, case.ε))
end

order(::WaveProblem{T,D,G,q}) where {T,D,G,q} = q
sampled(p::WaveProblem) = p.cfs !== nothing

"""
    prepare_rhs!(p::WaveProblem, u, t)

The first half of a right-hand-side evaluation: scatter `u` into the working
array and fill the ghosts, with the Dirichlet hook at time `t`. Split from
[`launch_rhs!`](@ref) so that an excision pass can run between the two.
"""
function prepare_rhs!(p::WaveProblem{T}, u, t) where {T}
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule; boundary=_dirichlet(p.case, p.fs.forest, t))
    return nothing
end

"""
    launch_rhs!(du, p::WaveProblem, t)

The second half: the right-hand-side kernel over every owned point, writing
`du` in state layout from the working array [`prepare_rhs!`](@ref) filled.
"""
function launch_rhs!(du, p::WaveProblem{T,D,G,q,DISS}, t) where {T,D,G,q,DISS}
    cwork = p.cfs === nothing ? nothing : p.cfs.work
    map_blocks!(wave_rhs_kernel!, p.fs, statearray(du, p.fs), p.fs.work, cwork,
                p.case.background, p.origins, p.spacings, convert(T, t), p.ε,
                Val(D), Val(G), Val(q), Val(DISS))
    return nothing
end

"""
    wave_rhs!(du, u, p::WaveProblem, t)

`du = F(u, t)`, in the SciML signature IMEXRungeKutta calls. `u` is
authoritative and never written; the working array is scratch, refreshed at
every call, so the right-hand side is a pure function of `(u, t)`. With
homogeneous boundary data (a periodic or reflecting domain) it is linear in
`u`, which is what an eigenvalue study of the semi-discrete operator needs.
"""
function wave_rhs!(du, u, p::WaveProblem, t)
    prepare_rhs!(p, u, t)
    launch_rhs!(du, p, t)
    return du
end

@kernel function speed_kernel!(λ, cwork, bg, @Const(origins), @Const(spacings), t,
                               ::Val{D}) where {D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    x = point_position(origins[b], spacings[b], I, Val(D))
    c = point_coefficients(cwork, bg, I, x, t, Val(D))
    s = characteristic_speed(c, 1)
    for d in 2:D
        s = max(s, characteristic_speed(c, d))
    end
    λ[ntuple(d -> I[d], Val(D))..., 1, b] = s
end

"""
    max_speed(p::WaveProblem, t) -> λ

The largest characteristic speed along any grid direction over the owned
points, `max_d (|β^d| + α√γ^{dd})`, at time `t` — what the time step is
sized by. On the field set's backend, through `mesh_mapreduce`.
"""
function max_speed(p::WaveProblem{T,D}, t) where {T,D}
    fs = p.fs
    λ = statevector(fs)
    cwork = p.cfs === nothing ? nothing : p.cfs.work
    map_blocks!(speed_kernel!, fs, statearray(λ, fs), cwork, p.case.background,
                p.origins, p.spacings, convert(T, t), Val(D))
    return mesh_mapreduce(identity, max, zero(T), fs, λ; vars=1:1)
end

"""
    convergence_rate(hs, errs) -> rate

The least-squares slope of `log(err)` against `log(h)`.
"""
function convergence_rate(hs, errs)
    x = log.(tofloat64.(hs))
    y = log.(tofloat64.(errs))
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end
