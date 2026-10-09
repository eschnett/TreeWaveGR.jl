# TreeWaveGR.jl

The scalar wave equation `□u = 0` on fixed, possibly time-dependent, curved
background spacetimes — static, spinning and moving Kerr–Schild black
holes, flat space in gauge-wave and superluminal charts, and model metrics —
second order in space and first order in time, on the adaptive octree of
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl). In 1, 2 and 3
dimensions, multi-threaded, on GPUs through KernelAbstractions, and in any
floating-point type.

It is the testbed for black-hole excision (TreeExcision): its `Π` equation
has the principal part of
[TreeGeneralizedHarmonic](https://github.com/eschnett/TreeGeneralizedHarmonic.jl)'s,
and every case it runs has an exact solution.

See [CODE.md](CODE.md) for the design.

## Example

```julia
using SpacetimeMetrics, TreeAMR, TreeWaveGR

# The static monopole on a Kerr hole (a = 0.6), on a box outside the horizon.
case = WaveCase(Background{3}(KerrSchild(1.0, 0.6)), StaticHole(1.0, 0.6);
                extents = ((2.5, 4.5), (2.5, 4.5), (-1.0, 1.0)))
r = wave_errors(Float64, case; N = 16, roots = 1, q = 4, t_end = 1/2)
```

## Testing

```julia
using Pkg
Pkg.test("TreeWaveGR")
```
