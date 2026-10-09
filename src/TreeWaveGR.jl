"""
    TreeWaveGR

The scalar wave equation `□u = 0` on a fixed, possibly time-dependent,
curved background spacetime, second order in space and first order in time,
on [TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl)'s adaptive octree of
uniform blocks. It is the testbed for TreeExcision: the same principal part
as TreeGeneralizedHarmonic's `Π` equation, on the Kerr–Schild holes —
static, spinning and moving — that excision has to handle.

See `CODE.md` for the design.
"""
module TreeWaveGR

using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using SpacetimeMetrics
using StaticArrays
using TreeAMR

import IMEXRungeKutta as IRK

include("precision.jl")
include("device.jl")
include("stencils.jl")
include("provider.jl")
include("backgrounds.jl")
include("exact.jl")
include("boundaries.jl")
include("coefficients.jl")

# Stencils and the provider seam
export derivative_weights, dissipation_weights, dissipation_rank
export StencilProvider, Centered

# Backgrounds and their coefficients
export Background, spacetime_metric, WaveCoefficients, ncoefficients, wave_coefficients
export characteristic_speed, ConstantShift, River
export coefficient_fieldset, fill_coefficients!

# Exact solutions
export inner_coordinates, base_chart, ExactSolution
export PlaneWave, PolynomialWave, StaticHole, StaticRiver
export exact_value, exact_state

# Boundaries
export even_parity, identity_rotation, has_outer_face, dirichlet

end # module TreeWaveGR
