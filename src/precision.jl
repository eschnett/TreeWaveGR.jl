# The operations this package performs that a *software* floating-point type
# does not provide, written so that they need only what every type here has.
#
# The package is generic in its element type `T`, and TreeAMR's mesh is too.
# What is not generic is `Base`: MultiFloats.jl implements `floor` and `ceil`
# returning a float, but no conversion to `Integer` and no `rem`, so
# `ceil(Int, x)` and `mod(x, y)` are `MethodError`s at `Float32x2` while
# working at `Float32` and `Float64`.
#
# Copied from TreeGeneralizedHarmonic (which copied it from TreeWave) rather
# than depended on: an application does not depend on another application.
#
# Nothing here is exported. They are spellings, not concepts.

"""
    wrap(x, L)

`x` reduced into `[0, L)` for positive `L` — what `mod(x, L)` means on a
periodic box. Spelled `x - L·floor(x/L)` because `Base.mod` on floats goes
through `rem`, which MultiFloats.jl does not define.
"""
wrap(x, L) = x - L * floor(x / L)

"""
    ceilint(x)
    floorint(x)

`ceil(Int, x)` and `floor(Int, x)`, for a type that may not define
`Int(::AbstractFloat)`. A software float goes through `BigFloat`, which every
`AbstractFloat` converts to; the value is already an exact integer there, so
the detour is exact. It allocates, so these are for host-side control flow
(step counts, chunk counts, buffer widths) and never per point.
"""
ceilint(x) = _toint(ceil(x))
floorint(x) = _toint(floor(x))

_toint(y::Base.IEEEFloat) = Int(y)
_toint(y) = Int(BigFloat(y))

"""
    tofloat64(x)

`x` as a `Float64`, for the analysis records. `Float64(::Float32x2)` is not
defined by MultiFloats.jl, so a software float goes through `BigFloat`.
Host-side, at chunk frequency, never per point.
"""
tofloat64(x::Base.IEEEFloat) = Float64(x)
tofloat64(x) = Float64(BigFloat(x))
