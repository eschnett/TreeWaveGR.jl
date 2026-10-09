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
include("evolution.jl")
include("stepping.jl")
include("initialdata.jl")

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

# The right-hand side and the integrator
export wave_rhs_point, WaveProblem, is_stationary, prepare_rhs!, launch_rhs!, wave_rhs!
export max_speed, convergence_rate
export state_partition, wave_integrator, wave_solve, wave_steps

# Cases
export WaveCase, with_dissipation, wave_forest, state_fieldset, wave_operators
export exact_callback, fill_exact!, exact_statevector, wave_errors

end # module TreeWaveGR
