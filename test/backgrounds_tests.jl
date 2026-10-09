# The backgrounds and their coefficients (`src/backgrounds.jl`,
# `src/coefficients.jl`).

# A spread of backgrounds in three dimensions: every SpacetimeMetrics wrapper
# this package knows the chart map of, and the two model metrics.
const BACKGROUNDS_3D = (
    Minkowski(),
    KerrSchild(1.0, 0.0),
    KerrSchild(1.0, 0.6),
    KerrSchild(1.0, 0.9),
    translate(KerrSchild(1.0, 0.6), [0.0, 0.5, -0.25, 0.125]),
    rotate(KerrSchild(1.0, 0.6), 0.3, 0.4, 0.5),
    boost(KerrSchild(1.0, 0.0), [0.3, 0.0, 0.0]),
    boost(KerrSchild(1.0, 0.0), [0.6, 0.0, 0.0]),
    boost(KerrSchild(1.0, 0.6), [0.0, 0.42, 0.42]),
    GaugeWave(0.1, 2.0),
    SineShift(0.2, 2.0),
    ShiftedMinkowski(0.4, 1.0),
    MovingGrid(1.5, 0.5, 0.0),
    ConstantShift([1.5, 0.25, -0.5]),
    River(1.0),
)

# Points well away from every hole above (the holes sit within a few M of
# the origin).
const POINTS_3D = ((3.5, 1.25, 2.0), (-2.75, 3.0, -1.5), (1.5, -2.5, 3.25))

@testset "Minkowski's coefficients are trivial: D=$D" for D in 1:3
    bg = Background{D}(Minkowski())
    c = wave_coefficients(bg, 0.25, ntuple(d -> 0.1d, D))
    @test c.αsγ == 1
    @test all(iszero, c.β)
    @test iszero(c.divβ)
    @test c.A == I
    @test all(iszero, c.divA)
    @test ncoefficients(D) == length(TreeWaveGR.pack_coefficients(c))
end

@testset "The coefficients agree with SpacetimeMetrics' ADM split: $(nameof(m))" for m in
                                                                                     BACKGROUNDS_3D
    bg = Background{3}(m)
    t = 0.375
    for x in POINTS_3D
        c = wave_coefficients(bg, t, x)
        α, β, γ = adm_decompose(m, SVector(t, x...))
        sγ = sqrt(det(γ))
        @test c.αsγ ≈ α / sγ rtol = 1e-13
        @test c.β ≈ β rtol = 1e-13 atol = 1e-14
        @test c.A ≈ α * sγ * inv(γ) rtol = 1e-13
        @test c.A == c.A'
        # The divergences by an independent forward-mode Jacobian, through
        # `adm_decompose` rather than this package's split.
        fβ(y) = adm_decompose(m, SVector(t, y...))[2]
        function fA(y)
            a, _, g = adm_decompose(m, SVector(t, y...))
            return vec(a * sqrt(det(g)) * inv(g))
        end
        Jβ = ForwardDiff.jacobian(fβ, SVector(x))
        JA = ForwardDiff.jacobian(fA, SVector(x))
        @test c.divβ ≈ tr(Jβ) rtol = 1e-12 atol = 1e-13
        for j in 1:3
            # `∂_i A^{ij}`: row `(i, j)` of `vec(A)` is `i + 3(j − 1)`.
            @test c.divA[j] ≈ sum(JA[i + 3(j - 1), i] for i in 1:3) rtol = 1e-12 atol = 1e-13
        end
        # The characteristic speed is `|β^d| + α √γ^{dd}`.
        for d in 1:3
            @test characteristic_speed(c, d) ≈ abs(β[d]) + α * sqrt(inv(γ)[d, d]) rtol = 1e-13
        end
    end
end

@testset "A slice is the leading block of the metric: D=$D" for D in 1:2
    m = River(1.0)
    bg = Background{D}(m)
    x = ntuple(d -> 2.5 + d, D)
    g = spacetime_metric(bg, 0.0, x)
    g4 = SpacetimeMetrics.metric(m, SVector(0.0, x..., ntuple(_ -> 0.0, 3 - D)...))
    @test g == g4[1:(D + 1), 1:(D + 1)]
    @test_throws ArgumentError Background{4}(m)
    @test_throws ArgumentError Background{0}(m)
end

@testset "The river has Kerr–Schild's inverse spatial metric, and outflow inside 2M" begin
    bg = Background{3}(River(1.0))
    for x in POINTS_3D
        g = spacetime_metric(bg, 0.0, x)
        gks = spacetime_metric(Background{3}(KerrSchild(1.0)), 0.0, x)
        @test inv(g)[2:4, 2:4] ≈ inv(gks)[2:4, 2:4] rtol = 1e-13
    end
    # At r = 1 < 2M along x, both characteristics along x point inward.
    c = wave_coefficients(bg, 0.0, (1.0, 0.0, 0.0))
    @test c.β[1] - sqrt(c.αsγ * c.A[1, 1]) > 0
end

@testset "Packing round-trips the coefficients exactly: D=$D" for D in 1:3
    bg = Background{D}(boost(KerrSchild(1.0, 0.6), [0.3, 0.2, 0.0]))
    c = wave_coefficients(bg, 0.25, ntuple(d -> 2.0 + d / 3, D))
    w = collect(TreeWaveGR.pack_coefficients(c))
    # Stored with a variable stride of 3, to check the stride is honoured.
    w3 = zeros(3 * length(w))
    w3[1:3:end] .= w
    @test TreeWaveGR.load_coefficients(w3, 1, 3, Val(D)) == c
end

@testset "The coefficients are allocation-free and inferred: D=$D" for D in 1:3
    bg = Background{D}(boost(KerrSchild(1.0, 0.6), [0.3, 0.0, 0.0]))
    x = ntuple(d -> 2.0 + d / 3, D)
    @inferred wave_coefficients(bg, 0.25, x)
    f(bg, x) = @allocated wave_coefficients(bg, 0.25, x)
    f(bg, x)
    @test f(bg, x) == 0
    @test isbits(wave_coefficients(bg, 0.25, x))
end

# Not bit for bit: the set is filled by a kernel and the comparison is
# evaluated on the host, and ForwardDiff's `muladd`s may fuse differently in
# the two compilations (observed in the last place of `divβ`).
@testset "The sampled set holds the analytic coefficients: D=$D" for D in 1:3
    forest = Forest{Float64}(ntuple(_ -> 2, D); N=8,
                             extents=ntuple(_ -> (2.0, 4.0), D))
    refine!(forest, [forest.leaves[1]])
    balance!(forest)
    bg = Background{D}(KerrSchild(1.0, 0.6))
    cfs = coefficient_fieldset(forest)
    @test cfs.nvars == ncoefficients(D)
    fill_coefficients!(cfs, bg, 0.0)
    st, sv, sb = TreeWaveGR.work_strides(cfs.work, Val(D))
    G0 = ntuple(_ -> 0, D)
    for b in 1:nblocks(cfs), I in CartesianIndices(ntuple(_ -> forest.N, D))
        Ib = (Tuple(I)..., b)
        x = coordinates(cfs, b, Tuple(I))
        c = TreeWaveGR.point_coefficients(cfs.work, bg, Ib, x, 0.0, Val(D))
        packed(c) = collect(TreeWaveGR.pack_coefficients(c))
        @test packed(c) ≈ packed(wave_coefficients(bg, 0.0, x)) rtol = 1e-14
        @test TreeWaveGR.point_coefficients(nothing, bg, Ib, x, 0.0, Val(D)) ==
              wave_coefficients(bg, 0.0, x)
    end
end
