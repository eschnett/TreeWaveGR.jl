# The chart maps and the exact solutions (`src/exact.jl`).

@testset "The chart map pulls the base metric back to the metric: $(nameof(m))" for m in
                                                                                    BACKGROUNDS_3D
    base = base_chart(m)
    for x in POINTS_3D, t in (0.0, 0.625)
        p = SVector(t, x...)
        J = ForwardDiff.jacobian(y -> inner_coordinates(m, y), p)
        ĝ = SpacetimeMetrics.metric(base, inner_coordinates(m, p))
        @test J' * ĝ * J ≈ SpacetimeMetrics.metric(m, p) rtol = 1e-13 atol = 1e-14
    end
end

# `∂_μ(√−g g^{μν} ∂_ν u)` at the spacetime point `z = (t, x_1, …, x_D)`, by
# nested forward-mode differentiation, independent of this package's
# coefficients and of `exact_state`.
function box_flux_divergence(sol, bg::Background{D}, z::SVector) where {D}
    u(y) = exact_value(sol, bg, y[1], SVector{D}(ntuple(i -> y[i + 1], D)))
    function F(y)
        g = spacetime_metric(bg, y[1], SVector{D}(ntuple(i -> y[i + 1], D)))
        return sqrt(-det(g)) * (inv(g) * ForwardDiff.gradient(u, y))
    end
    return tr(ForwardDiff.jacobian(F, z)), maximum(abs, F(z))
end

# Each exact solution on backgrounds that share its base chart, in the
# dimensions it is a solution in, at points outside every hole.
const SOLUTION_CASES = [
    (PlaneWave(1.0, [2.0, 0.0, 0.0]), 1, Minkowski()),
    (PlaneWave(1.0, [2.0, -1.0, 0.0], 0.3), 2, Minkowski()),
    (PlaneWave(0.5, [1.0, 2.0, -1.5], 0.3), 3, Minkowski()),
    (PlaneWave(1.0, [π, 0.0, 0.0]), 1, GaugeWave(0.1, 2.0)),
    (PlaneWave(1.0, [π, 1.0, 0.0]), 2, GaugeWave(0.1, 2.0)),
    (PlaneWave(1.0, [π, 1.0, 0.5]), 3, SineShift(0.2, 2.0)),
    (PlaneWave(1.0, [1.0, 1.0, 0.0]), 2, ShiftedMinkowski(0.4, 1.0)),
    (PlaneWave(1.0, [1.0, 0.5, 0.25]), 3, MovingGrid(1.5, 0.5, 0.0)),
    (PlaneWave(1.0, [1.0, 0.5, 0.0]), 2, ConstantShift([1.5, 0.25, 0.0])),
    (PlaneWave(1.0, [1.0, 0.5, 0.25]), 3, boost(Minkowski(), [0.6, 0.0, 0.0])),
    (PlanePulse(1.0, [1.0, 0.0, 0.0], 0.25, 0.5), 1, Minkowski()),
    (PlanePulse(1.0, [1.0, 1.0, 0.0], 0.25, 0.5), 2, boost(Minkowski(), [0.3, 0.2, 0.0])),
    (PlanePulse(1.0, [1.0, -1.0, 2.0], 0.5), 3, GaugeWave(0.1, 2.0)),
    (PolynomialWave(1.0, 2.0), 1, ConstantShift([1.5, 0.0, 0.0])),
    (PolynomialWave(1.0, 2.0), 3, ConstantShift([0.5, 0.25, 0.0])),
    (StaticHole(1.0, 0.0), 3, KerrSchild(1.0, 0.0)),
    (StaticHole(1.0, 0.6), 3, KerrSchild(1.0, 0.6)),
    (StaticHole(1.0, 0.9), 3, KerrSchild(1.0, 0.9)),
    (StaticHole(1.0, 0.6), 3, rotate(KerrSchild(1.0, 0.6), 0.3, 0.4, 0.5)),
    (StaticHole(1.0, 0.0), 3, boost(KerrSchild(1.0, 0.0), [0.3, 0.0, 0.0])),
    (StaticHole(1.0, 0.0), 3, boost(KerrSchild(1.0, 0.0), [0.6, 0.0, 0.0])),
    (StaticHole(1.0, 0.6), 3, boost(KerrSchild(1.0, 0.6), [0.6, 0.0, 0.0])),
    (StaticHole(1.0, 0.6), 3, translate(KerrSchild(1.0, 0.6), [0.0, 0.5, -0.25, 0.125])),
    (StaticRiver(1.0), 1, River(1.0)),
    (StaticRiver(1.0), 2, River(1.0)),
    (StaticRiver(1.0), 3, River(1.0)),
]

@testset "Every exact solution solves the wave equation: $(nameof(typeof(sol))), D=$D, $(nameof(m))" for (sol, D, m) in
                                                                                                        SOLUTION_CASES
    bg = Background{D}(m)
    TreeWaveGR.check_solution(sol, bg)
    for x in POINTS_3D, t in (0.0, 0.625)
        # Move the point outward in the first coordinates, away from every
        # hole (the river's horizon at D = 1 is a point at |x| = 2).
        z = SVector(t, ntuple(i -> x[i] + sign(x[i]) * 1.5, D)...)
        div, scale = box_flux_divergence(sol, bg, z)
        @test abs(div) <= 1e-12 * max(1, scale)
    end
end

@testset "The exact state's Π is √γ n^a ∂_a u: $(nameof(typeof(sol))), D=$D" for (sol, D, m) in
                                                                                  SOLUTION_CASES
    bg = Background{D}(m)
    for x in POINTS_3D
        xs = SVector{D}(ntuple(i -> x[i] + sign(x[i]) * 1.5, D))
        t = 0.375
        u, Π = exact_state(sol, bg, t, xs)
        @test u == exact_value(sol, bg, t, xs)
        ∇u = ForwardDiff.gradient(y -> exact_value(sol, bg, y[1], y[SVector{D}(2:(D + 1))]),
                                  SVector(t, xs...))
        α4, β4, γ4 = adm_decompose(spacetime_metric(Background{3}(m), t,
                                                    SVector(xs..., ntuple(_ -> 0.0, 3 - D)...)))
        # At D < 3 the slice's own split: the leading block of the metric.
        g = spacetime_metric(bg, t, xs)
        γ = g[2:(D + 1), 2:(D + 1)]
        β = γ \ g[2:(D + 1), 1]
        α = sqrt(-g[1, 1] + dot(β, g[2:(D + 1), 1]))
        @test Π ≈ sqrt(det(γ)) / α * (∇u[1] - dot(β, ∇u[2:end])) rtol = 1e-12 atol = 1e-13
        D == 3 && @test α ≈ α4 && β ≈ β4
    end
end

@testset "The exact state is allocation-free and inferred" begin
    bg = Background{3}(boost(KerrSchild(1.0, 0.6), [0.6, 0.0, 0.0]))
    sol = StaticHole(1.0, 0.6)
    x = (3.0, 1.0, 2.0)
    @inferred exact_state(sol, bg, 0.5, x)
    f(sol, bg, x) = @allocated exact_state(sol, bg, 0.5, x)
    f(sol, bg, x)
    @test f(sol, bg, x) == 0
end

@testset "A solution from another spacetime is refused" begin
    @test_throws ArgumentError TreeWaveGR.check_solution(StaticHole(1.0, 0.0),
                                                         Background{3}(Minkowski()))
    @test_throws ArgumentError TreeWaveGR.check_solution(PlaneWave(1.0, [1.0, 0, 0]),
                                                         Background{3}(KerrSchild(1.0)))
    @test_throws ArgumentError TreeWaveGR.check_solution(StaticHole(1.0, 0.5),
                                                         Background{3}(KerrSchild(1.0, 0.6)))
    @test_throws ArgumentError TreeWaveGR.check_solution(StaticHole(1.0, 0.0),
                                                         Background{2}(KerrSchild(1.0)))
    @test TreeWaveGR.check_solution(StaticHole(1.0, 0.0),
                                    Background{3}(boost(KerrSchild(1.0), [0.3, 0, 0]))) ===
          nothing
end
