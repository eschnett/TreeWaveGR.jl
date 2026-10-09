# Working notes for Claude in TreeWaveGR.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

## What this package is

The scalar wave equation on fixed curved backgrounds, on
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl): the testbed for
TreeExcision (`../TreeExcision`, `GOAL.md` there), with the principal part of
TreeGeneralizedHarmonic's `Π` equation. TreeAMR is the mesh;
SpacetimeMetrics supplies the backgrounds; this package is the equation.

Three rules:

- **No mesh machinery.** Trees, ghosts, interpolation and reductions belong
  in TreeAMR. If something is missing there, say so — the route is a TreeAMR
  release, not a workaround here.
- **No singularity handling.** That is TreeExcision's. What this package
  owns is the seam (`StencilProvider`, `wave_rhs_point`, the split
  `prepare_rhs!`/`launch_rhs!`); keep the main kernel's hot path free of
  excision logic.
- **Every test case has an exact solution.** A new background either comes
  with an `inner_coordinates` method (and passes the pullback test) or with a
  solution of its own (and passes the `□u = 0` test in `test/exact_tests.jl`).

## Commands

Run the tests (about 1.5 minutes, nearly all compilation):

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

With bounds checks (the RHS kernel is `@inbounds`) and threads — CI does
both:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes", "--threads=4"])'
```

`Pkg.test` does not inherit `-t`; the thread-independence test spawns its
own subprocess at another count either way.

Run on a GPU. No device package is a dependency, so this needs an
environment of its own — once, then reuse it:

```bash
julia --project=/tmp/twgrgpu -e 'using Pkg; Pkg.develop(path=".")
    Pkg.add(url="https://github.com/eschnett/SpacetimeMetrics.jl", rev="main")
    Pkg.add(["Metal", "KernelAbstractions", "MultiFloats", "TreeAMR", "Test",
             "StaticArrays", "ForwardDiff", "Random", "LinearAlgebra"])'
TREEWAVEGR_TEST_BACKEND=metal julia --project=/tmp/twgrgpu test/runtests.jl
julia --project=/tmp/twgrgpu bench/rhs.jl --backend=metal --type=f32
```

Unset, `TREEWAVEGR_TEST_BACKEND` runs `test/device_tests.jl` on `CPU()`.
Metal has no `Float64`, so the device tests run in `Float32` there.

## Things that will bite

- **The dependencies come from three places.** TreeAMR is in the General
  registry (`[compat]` `"0.2"`); SpacetimeMetrics, IMEXRungeKutta and
  TreeIOHDF5 are not, and are located by `[sources]` entries pointing at
  GitHub `main`. `Pkg.add` rewrites `Project.toml` and drops comments.
  The local checkouts in `~/src/jl/` are *not* what is tested.
- **Sampled and analytic coefficients agree to rounding, not bit for bit.**
  The fill kernel and the RHS kernel are different compilations, and
  ForwardDiff's `muladd`s fuse differently in them. Do not write a test that
  asks for `==` between them, or between a kernel and a host evaluation of
  `wave_coefficients`.
- **A slice is not a symmetry reduction.** `Background{D}` with `D < 3` takes
  the leading block of the 4-metric at zero dropped coordinates. An exact
  solution of the full metric is a solution on the slice only if nothing
  couples to a dropped axis: no shift, wave number or boost component along
  one. `WaveCase` checks the metric (`check_slice`); a `PlaneWave` with a
  wave number along a dropped axis it cannot see — that changes `ω` only.
- **Dirichlet on every ghost over-specifies the outgoing modes.** The RHS at
  the exact state converges at `q`; a long run converges at `q − 1`. Keep
  convergence runs short or periodic (`CODE.md`, "Boundaries").
- **RK4 is not exact on a polynomial solution** with time-dependent
  boundary data (stage order one). Compare precisions against a reference
  run, not against zero.
- **MultiFloats has no transcendentals.** Software-float tests run
  `ConstantShift` with `PolynomialWave`; `tofloat64` and `ceilint` are the
  conversions (`src/precision.jl`). `Float64(::Float32x2)` is a
  `MethodError`.
- **A case in `Float64` literals is retyped at every entry point**
  (`retype`). A new entry point that builds a run must call it, or a
  `Float32` kernel computes in `Float64` and Metal refuses to compile it
  ("unsupported use of double value"). A new metric or solution type needs
  an unparameterized constructor taking its fields, which `retype` rebuilds
  it with.
- **Closures that become kernel arguments capture only `isbits` values** —
  never a `Type` (use `convert(eltype(x), t)` / `oftype`), never a name
  assigned twice (a `Core.Box`).

## Conventions

- `TODO.md`, if it exists, is Erik's. Do not modify it.
- Spec first: a design change goes into `CODE.md` with the code.
- Copies from TreeGeneralizedHarmonic say so in their header, and keep its
  weights, summation orders and names, so that the two stay comparable.
