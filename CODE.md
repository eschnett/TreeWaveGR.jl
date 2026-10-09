# TreeWaveGR.jl — Design

TreeWaveGR solves the scalar wave equation `□u = 0` on a fixed, possibly
time-dependent, curved background spacetime. The equation is second order in
space and first order in time, on [TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl)'s
adaptive octree of uniform blocks.

It exists for [TreeExcision](../TreeExcision): a testbed for black-hole
excision, with the same principal part as TreeGeneralizedHarmonic's `Π`
equation, on the Kerr–Schild holes that excision has to handle. TreeExcision
lists the holes as a ladder: static, then spinning, then moving.

**Contents**

- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
- [Lineage](#lineage)
- [The equations](#the-equations)
- [Backgrounds](#backgrounds)
- [Coefficient sources](#coefficient-sources)
- [Exact solutions](#exact-solutions)
- [Discretization](#discretization)
- [The right-hand side](#the-right-hand-side)
- [Boundaries](#boundaries)
- [Time integration](#time-integration)
- [Cases](#cases)
- [Extension points](#extension-points)
- [Refinement and the driver](#refinement-and-the-driver)
- [Precision, threads, devices](#precision-threads-devices)
- [Measured results](#measured-results)
- [Milestones](#milestones)
- [Possible extensions](#possible-extensions)

## Goals

- **The scalar analogue of TreeGeneralizedHarmonic's `Π` equation.** It has
  the same densitized `Π`, the same expanded second-order operator, and the
  same stencils, ghost widths and interface operators. A closure designed and
  measured here should transfer to the generalized-harmonic system unchanged.
- **Any analytic background.** Any SpacetimeMetrics metric can serve as the
  background, including SpacetimeMetrics' boosted, translated and rotated Kerr–Schild holes, flat space in
  gauge-wave and superluminal charts, and two model metrics of this
  package's own. The background may depend on time.
- **An exact solution on every background a test runs.** Exact solutions
  supply the initial data, the Dirichlet data at every outer face, and the
  error measurement.
- **Every dimension `D = 1, 2, 3`.** TreeExcision's model problems
  are two-dimensional (its `notes/questions.md`, §2).
- **Threads, devices and software floats from the start.** TreeExcision's
  `GOAL.md` requires this. Test runs are expensive, and high-precision
  convergence studies run at `Float64x2`.

## Scope and non-goals

- **No mesh machinery.** Trees, ghost cells, interpolation and reductions
  belong upstream in TreeAMR.
- **No singularity handling.** The black hole's interior belongs to
  TreeExcision. What this package provides is the seam it plugs into (see
  [Extension points](#extension-points)). Until then, a case's domain must not
  contain a singular point, and every hole test here evolves a box outside the
  horizon.
- **No radiative boundary condition.** Every case has an exact solution, so
  every outer face takes Dirichlet data (see [Boundaries](#boundaries)).
- **Not conservative at coarse-fine faces.** The flux form exists, but the
  expanded form is what is discretized, as in TreeGeneralizedHarmonic.
  TreeAMR's interface restriction has nothing to restrict here.

## Lineage

- **TreeWave** is the flat-space scalar wave on TreeAMR: `∂_t u = v`,
  `∂_t v = ∇²u`. It is written against TreeAMR 0.1.1 and is periodic only. It is not a
  dependency, and it is not modified.
- **TreeGeneralizedHarmonic** (TreeGH) supplies the formulation
  (`notes/formulation.md` there), the stencils (`src/stencils.jl`), the
  kernel's linear indexing, the integrator (IMEXRungeKutta's RK4 with an
  ownership partition), the precision and device helpers, and the
  `StencilProvider` interface of its excision branch
  (`claude/launch-step-x8-668ad9`). The code is copied rather than depended
  on, because one application does not depend on another. The copies are
  generalized from three dimensions to `D`.
- **TreeHydro** is the most current consumer of TreeAMR's driver pattern.

## The equations

The variables are `u` and the densitized momentum
`Π = √γ n^a ∂_a u = (√γ/α)(∂_t u − β^i ∂_i u)`. This is TreeGH's `Π`, which it
inherited from the scalar testbed `WaveToySecondOrder`. With it, the flux
form `∂_μ(√−g g^{μν} ∂_ν u) = 0` becomes

```
∂_t u = β^i ∂_i u + (α/√γ) Π
∂_t Π = ∂_i(β^i Π + A^{ij} ∂_j u),        A^{ij} = α√γ γ^{ij}
```

because `√−g g^{tν} ∂_ν u = −Π` and `√−g g^{iν} ∂_ν u = β^i Π + A^{ij} ∂_j u`.
What is discretized is the expanded form, as in TreeGH:

```
∂_t Π = β^i ∂_i Π + (∂_i β^i) Π + A^{ij} ∂_i ∂_j u + (∂_i A^{ij}) ∂_j u
```

Kreiss–Oliger dissipation is added to both equations (see
[Discretization](#discretization)).

**Why this `Π`.** TreeExcision's `notes/questions.md` writes `Π = −n^a ∂_a u`,
"the same as TreeGH". TreeGH's `Π` is in fact densitized and has the opposite
sign, and the choice matters (decided with Erik, 2026-10-09):

- The densitized form needs no extrinsic curvature and no Christoffel
  symbols. It needs only first derivatives of the metric.
- Its energy `½∫[Π²/√γ + √γ γ^{ij} ∂_i u ∂_j u]` is positive for any shift,
  superluminal included, wherever `α > 0` (TreeGH formulation.md).
- Its principal part is TreeGH's, term for term, so excision closures carry
  over.

**The coefficients.** The background enters only through five fields,
`WaveCoefficients`. That is `2 + 2D + D(D+1)/2` numbers per point (14 at
`D = 3`):

| field | value |
|---|---|
| `αsγ` | `α/√γ` |
| `β` | `β^i` |
| `divβ` | `∂_i β^i` |
| `A` | `α√γ γ^{ij}`, symmetric |
| `divA` | `∂_i A^{ij}`, one per `j` |

**Characteristic speeds.** Along grid direction `d` they are
`−β^d ± α√(γ^{dd})`, and `α²γ^{dd} = αsγ · A^{dd}`. Both have the same sign,
so the grid line is outflow at one end, exactly where `g^{dd} < 0`. That is
the criterion TreeExcision's grazing-line analysis is about.
`characteristic_speed(c, d)` returns `|β^d| + α√(γ^{dd})`, which sizes the
time step.

## Backgrounds

`Background{D}(m)` wraps any `SpacetimeMetrics.AbstractMetric`.

- **At `D = 3`** it is `m` itself.
- **At `D < 3`** it is the `(D+1)`-dimensional spacetime whose metric is the
  leading `(D+1)×(D+1)` block of `m`, evaluated where the dropped
  coordinates are zero. This is a *model* spacetime, not a symmetry
  reduction:
  - The slice of Kerr–Schild at `z = 0` is not Kerr.
  - The slice of the river metric is TreeExcision's 2D model.
  - `WaveCase` refuses a slice whose metric couples the evolved coordinates to
    a dropped one (`check_slice`). The simplest example is a constant shift
    with a component along a dropped axis, under which an exact solution of
    the full metric is not a solution on the slice.

`wave_coefficients(bg, t, x)` computes the coefficients. It forms
`(αsγ, β, A)` by the 3+1 split of the metric, and their divergences by **one**
forward-mode pass with `D` partials through that split. The metric is
evaluated once, on dual numbers. The computation is allocation-free and
`isbits`, so it runs in a kernel. `A` is symmetrized exactly, since `inv` is
symmetric only to the last place and a sampled set stores one triangle.

This package defines two model metrics:

- **`ConstantShift(β)`** is flat space with a uniform shift, i.e. Minkowski in the
  chart `x̂ = x + βt`. A superluminal `β` makes one face pure outflow. This is
  TreeExcision's model problem 1.
- **`River(M)`** has unit lapse, a flat spatial metric, and
  `β^i = √(2M/r) x^i/r`. At `D = 3` it is Schwarzschild in
  Painlevé–Gullstrand coordinates, with Kerr–Schild's `g^{ij}`. At `D = 2` it
  is TreeExcision's model problem 2.

## Coefficient sources

The kernel takes the coefficients from one of two sources. Which one is
chosen by the type of a single kernel argument:

- **sampled** (`cwork` is the coefficient field set's working array). This set
  is vertex-centered with `G = 0`, so it is never ghost-filled. It is filled
  once per mesh by `fill_by_coordinates!`. The right-hand side then does no
  metric algebra. It is allowed only for a background that `is_stationary`.
- **analytic** (`cwork === nothing`). `wave_coefficients` runs in the kernel
  at the point and the stage time. Time-dependent backgrounds such as boosted
  holes and the gauge wave need this source.

`WaveProblem(…; coefficients = :auto)` takes the sampled source where it is
allowed. The kernel forms the point's position by TreeAMR's own expression,
so both sources evaluate the coefficients at the same positions. They agree
to rounding, **not bit for bit**. The sampled set is filled by another kernel,
and ForwardDiff's `muladd`s fuse differently in the two compilations; this
was observed in the last place of `∂_i β^i`.

## Exact solutions

A scalar transforms trivially under a change of chart. If `f` solves the
equation in the chart `x̂`, then `u(x) = f(x̂(x))` solves it in `x`.

SpacetimeMetrics builds its test backgrounds as pullbacks of an inner metric
through a chart map. `inner_coordinates(m, x)` spells each map out once more:

| type | `x̂(x)` |
|---|---|
| `TranslatedMetric` | `x − d` |
| `RotatedMetric` | `Rᵀx` |
| `BoostedMetric` | `Λᵀx` |
| `GaugeWaveMetric` | `t − C cos φ`, `x + C cos φ` |
| `SineShiftMetric` | `x + C sin φ` |
| `ShiftedMinkowskiMetric` | `t + ψ(x)` |
| `MovingGridMetric` | `x + tV(x)` |
| `ConstantShift` | `x + βt` |

The maps are applied recursively down to the base chart (`base_chart`). The
test `J ĝ Jᵀ = g` holds each map to the metric it claims to pull back.

The solutions, each defined in a base chart:

| solution | chart | `u` | dimensions |
|---|---|---|---|
| `PlaneWave(A, k, φ)` | Minkowski | `A sin(k·x̂ − |k| t̂ + φ)` | any |
| `PlanePulse(A, n, w, s)` | Minkowski | `A exp(−((n·x̂ − t̂ − s)/w)²)` | any |
| `PolynomialWave(A, L)` | Minkowski | `A(t̂x̂ + (t̂² + x̂²)/2)/L²` | any |
| `StaticHole(M, a)` | Kerr–Schild | `ln((r − r₊)/(r − r₋))` | 3 |
| `StaticRiver(M)` | river | the radial monopole (see its docstring) | any |

`PolynomialWave` needs no transcendental function, so it is the solution
used in the software-float tests.

`StaticHole` is a function of Boyer–Lindquist `r` alone, which equals
Kerr–Schild's `r`. It solves `∂_r(Δ∂_r u) = 0`. Kerr–Schild's `t` and `φ`
differ from Boyer–Lindquist's by functions of `r`, which a stationary
axisymmetric `u` does not see. It is singular on the horizon, so a hole case
evolves a box outside `r₊`.

Wrapped in `boost`, it is an exact, time-dependent solution on a moving
hole. That covers the scalar half of every TreeExcision success case: a = 0,
0.6 and 0.9; v = 0.3 and 0.6; and a = 0.6 with v = 0.6. It is also not
trivial on a static hole: `Π = −(√γ/α) β·∇u ≠ 0` in Kerr–Schild, so the
advection terms are exercised.

`exact_state(sol, bg, t, x)` returns `(u, Π)`. It takes one forward-mode pass
with `D + 1` partials through the chart map and the solution, and is
allocation-free. `check_solution` refuses a solution from another spacetime,
or with other parameters.

The test suite checks `□u = 0` for every solution on every background it is
paired with, by nested automatic differentiation of the flux form. The check
is independent of the coefficients and of `exact_state`.

## Discretization

The scheme follows TreeGH's "Discretization".

- **Centered finite differences** of even order `q`, for `∂_i` and `∂_i∂_i`.
  The weights are built in `Rational{BigInt}` and emitted as `T(num)/T(den)`
  in the caller's world.
- **`∂_i∂_j` (`i ≠ j`)** is the tensor product of two first-derivative
  stencils. The outer sum runs along `i` and the inner along `j`. It reads
  the edge and corner ghosts that TreeAMR fills unconditionally.
- **Kreiss–Oliger dissipation** of rank `r = q/2 + 1` is applied to both
  variables along every axis, scaled by `ε/h`. It damps Nyquist at exactly
  `ε/h`, and it is a case parameter.
  - Wherever `g^{dd} < 0`, a second-order-in-space scheme needs it
    (Calabrese 2004).
  - Across a coarse-fine face with a superluminal shift it is needed here too
    (see [Measured results](#measured-results)).
- **Layout.** Vertex centering. Ghost width `G = q/2 + 1`, since the
  dissipation reaches one point past the derivatives. Point-value
  prolongation and restriction are of order `q + 2`, two above the
  differencing, so a second derivative keeps order `q` across a coarse-fine
  face (`wave_operators(q)`). TreeAMR requires `N ≥ 2G + 2`.
- **Advection is centered.** The provider's `adv` hook is where an upwinded
  or lopsided derivative would go.

## The right-hand side

Each `wave_rhs!(du, u, p, t)` does three things, which is TreeAMR's
contract:

    scatter!(fs, u)                                   # prepare_rhs!
    fill_ghosts!(fs, schedule; boundary = dirichlet(case, forest, t))
    map_blocks!(wave_rhs_kernel!, …)                  # launch_rhs!

- `u` is never written.
- The working array is scratch, so the right-hand side is a pure function of
  `(u, t)`.
- With homogeneous boundary data (a periodic or reflecting domain) the
  right-hand side is linear in `u`, which is what an eigenvalue study of the
  semi-discrete operator needs (TreeExcision's `notes/questions.md`, §3).

`wave_rhs_kernel!` runs once per owned point. It does the following:

1. Forms the position by TreeAMR's expression.
2. Indexes the working array linearly: a base index and one stride per axis,
   which TreeGH measured to be 2.4× faster than cartesian indexing.
3. Takes the coefficients from the source.
4. Calls `wave_rhs_point(S, c, work, bu, bΠ, inv_h, εh, Val(D), Val(DISS))`
   with `S = Centered(T, Val(q), strides)`.

`wave_rhs_point` is the pointwise operator. It sums in a fixed order:

1. Per axis: advection, the divergence terms, and the diagonal second
   derivatives.
2. The mixed derivatives `i < j`, row by row.
3. The dissipation.

The order is part of the operator.

`WaveProblem` carries the field set, the schedule, the case, the coefficient
set, and the per-block geometry uploaded to the field set's backend. It
holds `D`, `G`, `q` and whether dissipation is on as type parameters, so the
kernel's `Val`s are compile-time constants. A problem belongs to one mesh, so
a new one is built after a regrid.

## Boundaries

- **Periodic and reflecting faces, and the rotating seam,** are properties
  of the forest, and TreeAMR fills them. `u` and `Π` are scalars, so they are
  `EvenParity` under every mirror and turn into themselves under the quarter
  turn (`even_parity`, `identity_rotation`). Auxiliary `G = 0` sets declare
  the same; nothing reads it.
- **Every other face is outer.** `dirichlet(case, forest, t)` is a TreeAMR
  `CellBoundary` that sets every outer ghost point, including edges, corners
  and the vertex-centered high plane, to the exact state at time `t`. It runs
  as a kernel on any backend. It goes to every ghost fill, to `regrid!`, and
  to `adapt_to_initial_data!`.

Dirichlet data on *all* ghost values over-specifies the outgoing
characteristic.

- With exact data this is consistent, and the right-hand side converges at
  order `q`.
- In a run, error that reaches such a face meets the exact ghost values. The
  result converges at `q − 1` asymptotically. Measured:
  - on a superluminal outflow face, 3.0 at `q = 4` and 1.06 at `q = 2`;
  - on a timelike face at fine resolution, 3.5 and falling;
  - on the holes, see [Measured results](#measured-results).

  Two mechanisms contribute. At `q ≥ 4` the stencil reaches two ghost
  layers, and exact data on both is more than the error equation's one
  boundary condition. The advection term `β^d ∂_d Π` reads `Π`'s ghosts,
  which imposes the outgoing `Π` as well.

Convergence tests therefore run short, or periodic. An outflow face is
precisely what excision has to close without data, and that is
TreeExcision's subject.

## Time integration

IMEXRungeKutta's explicit `RK4` with fixed steps (`wave_integrator`,
`wave_solve`). The stage arithmetic runs by block owner through
`state_partition`, so a run is bit-identical at every thread count.

`wave_steps(p, t0, t1; cfl = 1/4)` sizes the step by
`dt ≤ cfl · h_min / λ`, where `λ = max_speed(p, t0)` is the largest
characteristic speed over the owned points. `max_speed` is a kernel plus
`mesh_mapreduce`.

RK4's stage order is one. A solution whose boundary data depend on time is
therefore not integrated exactly, even when it is polynomial in `t`.

## Cases

`WaveCase(background, solution; extents, periodic, reflecting, rotating, ε)`
is an `isbits` description of one run. The helpers built on it are:

- `wave_forest(T, case; N, roots, refined)`: a forest of uniform roots. With
  `refined = true` it is a frozen two-level hierarchy, with the first root
  refined.
- `state_fieldset(forest, q)`
- `exact_callback`, `fill_exact!` and `exact_statevector`
- `wave_errors(T, case; N, roots, q, t_end)`: one fixed-step run and its
  volume-weighted `L2`/`L∞` errors. Every convergence test is built from it.

## Extension points

TreeExcision plugs into the following, none of which is on the main kernel's
hot path:

1. **`StencilProvider`.** Five methods answer the stencil questions at one
   point: `d1`, `d2`, `dmix`, `ko` and `adv`. This is the interface of
   TreeGH's excision branch, made `D`-generic. `Centered` is the main
   kernel's provider. An excision provider stops its stencils at the
   excision surface and blends `adv` by the sign of `β^d`.
2. **`wave_rhs_point`**, the pointwise operator generic over the provider.
   An excision zone kernel calls it with its own provider at the points next
   to the surface.
3. **`prepare_rhs!` / `launch_rhs!`.** The right-hand side is split after the
   ghost fill, so that an excision pass can run between the two halves.
4. **`wave_coefficients` and `characteristic_speed`.** These are the shift and
   the characteristic speeds per grid direction, which the outflow criterion
   and the grazing-line census read.
5. **The right-hand side as an operator.** With homogeneous data it is
   linear, so a matrix-free Arnoldi iteration can find the rightmost
   eigenvalues of the semi-discrete system.

## Refinement and the driver

**The indicator** (`src/refinement.jl`) is Löhner's, on `u`, with the
noise floor referred to the global amplitude `max |u|`:

    τ = |u₊ − 2u₀ + u₋| / (|u₊ − u₀| + |u₀ − u₋| + 4ε·max|u|)

It takes the maximum over the dimensions.

- **Why a global floor.** It is TreeWave's lesson. `u` crosses zero, and a
  floor that scales with the local values refines numerical dust.
- **How the flags are formed.** TreeAMR's `firing_boxes` evaluates `τ` per
  point on the field set's backend, twice, and turns it into one flag per
  block. The flag is:
  - `(Refine, box)` above `refine_tol` below the cap;
  - `(Keep, box)` above `coarsen_tol`;
  - `Coarsen` otherwise.

  The gap between the two tolerances is the dead band. They have no defaults.
- **A level floor.** A ball, possibly moving at a constant velocity, holds
  every block that reaches into it at a given level whatever `τ` says. A
  hole case wants resolution at the horizon before anything there is a
  feature, and TreeExcision's surface needs it.
- **The regrid's margin** is `refinement_buffer`: the distance the fastest
  characteristic travels in one chunk, in cells at the cap, plus one. It is
  measured on the unrefined forest at `t = 0`, so a restart derives the same
  margin. It may not exceed `N`, since TreeAMR's recruitment reaches one ring
  of neighbours.

**The driver.** `evolve!(T, case; N, roots, q, t_end, chunk, refinement, …)`
is TreeGH's and TreeHydro's chunk loop:

- **The initial mesh** is adapted to the exact initial data by
  `adapt_to_initial_data!`, which re-evaluates the data on every new mesh.
- **Each chunk:**
  1. A fixed number of RK4 steps on one mesh, sized by the speed at the
     chunk's start. The integrator's scratch is reused while the mesh
     stands.
  2. A record: the state's L2/L∞ error against the exact solution, its norm,
     the step and block counts and the finest level, all as `Float64`.
  3. The observer.
  4. A regrid. This means `scatter!`, a ghost fill with the Dirichlet hook,
     the flags, and `regrid!` with the hook. After a move, the schedule, the
     state vector and the problem (with its sampled coefficients) are
     rebuilt.
  5. A checkpoint.
- **Chunk end times** are `c · chunk`, or `t_end` for the last, computed and
  never accumulated.

**Checkpoints** (`src/checkpoint.jl`) go through TreeIOHDF5's
`save_checkpoint`/`load_checkpoint`.

- **What is stored.** The mesh, the state vector, the chunk index, and the
  step and regrid counts.
- **When.** After the regrid, where a fixed-step integrator holds nothing
  but `(t, u)`. `t` is a function of the chunk index.
- **Restart.** `restart = path` is the same `evolve!` call continued, and the
  order and the block size are checked against the file. It is bit for bit:
  the restarted run's leaves, state, step count and records are those of the
  run that did not stop (`test/driver_tests.jl`).

## Precision, threads, devices

- **Precision.** Generic in `T`. There are no float literals in per-point
  arithmetic: weights are emitted as `T(num)/T(den)`, offsets are `one(h)`,
  and conversions are explicit. The suite runs `Float32`, `Float64`, and
  `Float32x2`, which is `BigFloat`'s twin for this purpose. MultiFloats has no
  transcendental functions, so the software type runs the algebraic case:
  `ConstantShift` and `PolynomialWave`. Each type reproduces a `BigFloat`
  run of the same case to its own precision. The host-side spellings
  `ceilint`, `floorint` and `tofloat64` are in `src/precision.jl`.
- **Threads.** There are no loops of this package's own. TreeAMR's kernels
  thread themselves, and IMEXRungeKutta combines stages by owner. The
  thread-independence test runs `test/thread_workload.jl` in a subprocess at
  another thread count and requires identical digests.
- **Devices.** Every per-point pass is a KernelAbstractions kernel on the
  field set's backend:
  - the right-hand side;
  - the speed;
  - the coefficient fill;
  - the exact state, as `fill_by_coordinates!` and as the Dirichlet hook.

  Callbacks capture only `isbits` values. This includes the analytic
  coefficients, so SpacetimeMetrics' metric and ForwardDiff's dual pass
  through it run in the kernel. A case is written in convenient literals;
  `retype(T, case)` rebuilds its background, solution and domain in the
  run's type at every entry point. A `Float64` parameter in a `Float32`
  kernel would compute in `Float64`, which Metal refuses to compile.
  `test/device_tests.jl` runs the right-hand side, a solve and an adaptive
  run on the device and compares them with the host. The device runs give
  the same mesh, and the same states to the type's precision. Set
  `TREEWAVEGR_TEST_BACKEND=metal` or `cuda` to run it on a device; it runs
  on `CPU()` otherwise.

## Measured results

All on an Apple M3 Pro, `Float64`.

- **The right-hand side converges at order `q`** at the exact state, against
  a sixth-order difference in time. This holds for `q = 2` and `4` on every
  background kind, including the superluminal ones and the boosted hole. On
  the frozen two-level hierarchy it holds at `D = 1, 2, 3`
  (`test/evolution_tests.jl`).
- **Runs converge at order `q`** on periodic domains: Minkowski at `D = 1, 2,
  3` and the gauge wave at `D = 2`.
- **TreeExcision's ladder, on the box `[3, 5]×[−1, 1]²`** outside the
  horizon. The right-hand side at the exact state converges at `q` on every
  rung (`q = 2`: 2.1–2.3; `q = 4`: 4.4–4.7; `N = 8, 12, 16`). Runs to
  `t = 1/4` at `q = 4`, `N = 16, 24, 32`, measured on the L2 error:

  | case | L2 error at N = 32 | rate |
  |---|---|---|
  | a = 0 | 1.1e-6 | 3.64 |
  | a = 0.6 | 6.5e-7 | 3.62 |
  | a = 0.9 | 2.5e-7 | 3.59 |
  | v = 0.3 | 1.2e-6 | 3.14 |
  | v = 0.6 | 5.8e-7 | 2.66 |
  | a = 0.6, v = 0.6 | 4.3e-7 | 2.66 |

  - The rates are below `q` because Dirichlet data on every ghost
    over-specify the outgoing modes (see [Boundaries](#boundaries)). They
    tend to `q − 1`: the static `a = 0.6` hole runs at 2.9 at `N = 8, 12, 16`
    and at 3.6 at `N = 16, 24, 32`. Shrinking the step by 16 changes nothing,
    so this is spatial.
  - At `q = 2` the same runs give 1.6 on the static holes and 0.9 on the
    moving ones.
  - The advection term reads `Π`'s ghosts, so a larger shift through the
    faces costs more.
- **The river at `D = 2`** (`[3, 5]×[−1, 1]`, `t = 1/2`, `N = 16, 32, 64`)
  converges at 4.0.
- **Tracking a pulse.** A Gaussian plane pulse crosses the unit square
  diagonally at `D = 2`, on 4² roots of `N = 8`, two levels deep, to `t = 1/4`
  in chunks of `1/32`. The adaptive run's L2 error is the uniform run's at
  its finest resolution (2.354e-4 against 2.354e-4) on at most 187 blocks of
  the 256 that resolution needs everywhere.
- **The cost of a right-hand side** (`bench/rhs.jl`): the hole box at
  `q = 4` with dissipation, `N = 16`, 4³ roots (262 144 points), in ns per
  point:

  | | 1 thread `Float64` (32 768 points) | 8 threads `Float64` | Metal `Float32` |
  |---|---|---|---|
  | static hole, sampled: kernel | 27.1 | 7.1 | 1.5 |
  | static hole, sampled: `wave_rhs!` | 83.7 | 22.2 | 7.0 |
  | static hole, analytic: kernel | 141.7 | 29.9 | 3.6 |
  | boosted hole, analytic: kernel | 184.9 | 38.2 | 3.9 |
  | boosted hole, analytic: `wave_rhs!` | 300.0 | 58.8 | 9.2 |

  - **The ghost fill dominates** the sampled right-hand side on the CPU.
  - **The analytic coefficients cost 5× the sampled ones on the CPU** — a
    dual pass through the metric at every point — and 2.4× on the GPU,
    where the stencils' memory traffic hides more of it. This is why
    `:auto` samples a stationary background.
  - **No SIMD lanes or performance tuning yet.** TreeExcision's `GOAL.md`
    defers both.
- **Symmetric domains.** A reflecting face at `z = 0`, and the rotating
  quadrant above the hole, each reproduce the full box to `1e-11` relative
  after four steps with dissipation. The full box is itself symmetric only
  to its truncation error, because a vertex-centered block evolves its low
  boundary plane while the hook fills its high one.
- **Noise and dissipation.** Noise of `1e-8` was put on a quiet state on a
  periodic `D = 2` mesh, and the first unit of time was discarded as a
  transient. Over the next three time units the noise grows only on the
  two-level mesh with the superluminal shift `β = (1.5, 0.5)`, and there
  `ε = 0.5` cures it:

  | shift | mesh | `ε = 0` | `ε = 0.5` |
  |---|---|---|---|
  | `(0.5, 0)` | uniform | 0.90 | 0.44 |
  | `(0.5, 0)` | two-level | 0.99 | 0.50 |
  | `(1.5, 0.5)` | uniform | 0.92 | 0.44 |
  | `(1.5, 0.5)` | two-level | **3.4** | 0.54 |

  The numbers are the norm at `t = 4` relative to `t = 1` (N = 16, two
  roots per edge, `q = 4`); the test asserts the same over `t = 1 … 3`. So
  the coarse-fine face is where a large shift needs dissipation, and a hole
  run with refinement around it will need `ε > 0`.

## Milestones

| | milestone | status |
|---|---|---|
| W0 | skeleton, stencils, provider | done |
| W1 | backgrounds, coefficients, exact solutions | done |
| W2 | right-hand side, integrator, flat and curved convergence, threads, types | done |
| W3 | hole cases (a = 0, 0.6, 0.9; v = 0.3, 0.6; a = 0.6 with v = 0.6), river, reflecting and rotating domains | done |
| W4 | refinement criterion, driver, checkpoint and restart (TreeIOHDF5) | done |
| W5 | device tests, benchmark | done |

## Possible extensions

- **A radiative outer boundary.** TreeAMR's interior-reading boundary hook
  runs on the host only.
- **Upwinded advection** as a provider.
- **SIMD lanes on the CPU**, as in TreeGH's `src/lanes.jl`. TreeExcision's
  goal defers SIMD and a detailed performance analysis.
- **A Sommerfeld-free scattering problem** on a hole, once TreeExcision
  closes the interior. Quasinormal ringing is the physics check that this
  package's tests cannot make without one.
