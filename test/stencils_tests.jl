# The weights and the kernel-side contractions (`src/stencils.jl`) and the
# centered provider (`src/provider.jl`).

const STENCIL_ORDERS = (2, 4, 6, 8)

@testset "The weights are the textbook tables" begin
    R = Rational{BigInt}
    @test TreeWaveGR.rational_derivative_weights(2, 1) == R[-1//2, 0, 1//2]
    @test TreeWaveGR.rational_derivative_weights(2, 2) == R[1, -2, 1]
    @test TreeWaveGR.rational_derivative_weights(4, 1) == R[1//12, -2//3, 0, 2//3, -1//12]
    @test TreeWaveGR.rational_derivative_weights(4, 2) == R[-1//12, 4//3, -5//2, 4//3, -1//12]
    @test TreeWaveGR.rational_dissipation_weights(2) == R[-1, 4, -6, 4, -1] .// 16
    @test TreeWaveGR.rational_dissipation_weights(3) == R[1, -6, 15, -20, 15, -6, 1] .// 64
end

@testset "Each derivative is exact to its degree and not one further: q=$q" for q in
                                                                               STENCIL_ORDERS
    w1 = TreeWaveGR.rational_derivative_weights(q, 1)
    w2 = TreeWaveGR.rational_derivative_weights(q, 2)
    r = q ÷ 2
    js = big.(-r:r)
    # `∂^m x^p` at 0 is `m!` for `p == m` and zero otherwise.
    for p in 0:q
        @test sum(w1 .* js .^ p) == (p == 1 ? 1 : 0)
    end
    @test sum(w1 .* js .^ (q + 1)) != 0
    for p in 0:(q + 1)
        @test sum(w2 .* js .^ p) == (p == 2 ? 2 : 0)
    end
    @test sum(w2 .* js .^ (q + 2)) != 0
end

@testset "The dissipation annihilates low degrees and damps Nyquist: q=$q" for q in
                                                                             STENCIL_ORDERS
    r = q ÷ 2 + 1
    @test dissipation_rank(Val(q)) === Val(r)
    w = TreeWaveGR.rational_dissipation_weights(r)
    js = big.(-r:r)
    for p in 0:(2r - 1)
        @test sum(w .* js .^ p) == 0
    end
    # On the Nyquist mode `(-1)^j` the contraction is exactly `-1`.
    @test sum(w .* (-1) .^ js) == -1
end

@testset "The weights are isbits, inferred and rounded once: T=$T" for T in
                                                                     (Float32, Float64,
                                                                      Float32x2)
    for q in STENCIL_ORDERS
        w = @inferred derivative_weights(T, Val(q), Val(2))
        @test isbits(w)
        @test eltype(w) === T
        exact = TreeWaveGR.rational_derivative_weights(q, 2)
        @test all(w[k] == T(numerator(exact[k])) / T(denominator(exact[k]))
                  for k in eachindex(w))
        @test @inferred(dissipation_weights(T, dissipation_rank(Val(q)))) isa SVector
    end
end

@testset "The kernel contractions are the reference contractions: D=$D, q=$q" for D in
                                                                                  1:3,
                                                                              q in (2, 4)
    T = Float64
    rng = Random.MersenneTwister(D + q)
    n = 2q + 3
    nv, nb = 2, 2
    work = randn(rng, T, ntuple(_ -> n, D)..., nv, nb)
    st, sv, sb = TreeWaveGR.work_strides(work, Val(D))
    @test st == strides(work)[1:D]
    @test sv == strides(work)[D + 1]
    @test sb == strides(work)[D + 2]
    c = n ÷ 2 + 1
    I = (ntuple(_ -> c, D)..., 2)
    G = ntuple(_ -> 0, D)
    base = TreeWaveGR.linear_index(I, G, st, sb, Val(D)) + sv  # variable 2
    @test work[base] == work[ntuple(_ -> c, D)..., 2, 2]
    S = Centered(T, Val(q), st)
    w1 = derivative_weights(T, Val(q), Val(1))
    w2 = derivative_weights(T, Val(q), Val(2))
    wk = dissipation_weights(T, dissipation_rank(Val(q)))
    line(d) = [work[ntuple(e -> e == d ? i : c, D)..., 2, 2] for i in 1:n]
    for d in 1:D
        @test TreeWaveGR.d1(S, work, base, d) == TreeWaveGR.apply_stencil(w1, line(d), c)
        @test TreeWaveGR.d2(S, work, base, d) == TreeWaveGR.apply_stencil(w2, line(d), c)
        @test TreeWaveGR.ko(S, work, base, d) == TreeWaveGR.apply_stencil(wk, line(d), c)
        @test TreeWaveGR.adv(S, 1.5, 0.25, work, base, d) == 0.25
    end
    for i in 1:D, j in (i + 1):D
        f(a, b) = work[ntuple(e -> e == i ? c + Int(a) : e == j ? c + Int(b) : c, D)..., 2, 2]
        @test TreeWaveGR.dmix(S, work, base, i, j) ==
              TreeWaveGR.apply_mixed_stencil(w1, f, 0, 0, 1, 1)
    end
end

@testset "The operators converge at their order on smooth data: q=$q" for q in (2, 4, 6)
    f(x) = sin(x) * exp(x / 3)
    f1(x) = cos(x) * exp(x / 3) + sin(x) * exp(x / 3) / 3
    f2(x) = (-sin(x) + 2cos(x) / 3 + sin(x) / 9) * exp(x / 3)
    errs1, errs2 = Float64[], Float64[]
    hs = [1 / 8, 1 / 16, 1 / 32]
    for h in hs
        w1 = derivative_weights(BigFloat, Val(q), Val(1))
        w2 = derivative_weights(BigFloat, Val(q), Val(2))
        x = big(7) / 10
        push!(errs1, Float64(abs(TreeWaveGR.apply_stencil(w1, f, x, h) / h - f1(x))))
        push!(errs2, Float64(abs(TreeWaveGR.apply_stencil(w2, f, x, h) / h^2 - f2(x))))
    end
    @test log2(errs1[2] / errs1[3]) ≈ q atol = 0.2
    @test log2(errs2[2] / errs2[3]) ≈ q atol = 0.2
end

@testset "A stencil that does not exist is refused" begin
    @test_throws ArgumentError derivative_weights(Float64, Val(3), Val(1))
    @test_throws ArgumentError derivative_weights(Float64, Val(4), Val(3))
    @test_throws ArgumentError dissipation_weights(Float64, Val(0))
end
