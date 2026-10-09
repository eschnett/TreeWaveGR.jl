# Where the work runs.
#
# TreeAMR's M6 makes the *storage* decide: a backend keyword on `FieldSet`
# and `GhostSchedule` and nowhere else, after which `statevector` allocates
# where the field set lives, `regrid!` reallocates there, and every kernel
# takes its backend from the array it is handed. So an application that
# already goes through `map_blocks!` and `fill_by_coordinates!` is most of
# the way there, and what is left is only the places where *this* package
# keeps something of its own that a kernel reads.
#
# There are two such places, and this file is both of them: per-block
# metadata the right-hand side reads (`WaveProblem`'s origins and spacings),
# and the reverse direction, bringing a field set back to the host so
# something that can only run there -- a plot, a test's point-by-point
# comparison -- can read it.
#
# Copied from TreeGeneralizedHarmonic rather than depended on, as
# `precision.jl` is.

"""
    to_backend(backend, a)

`a` on `backend`: itself on the CPU, a fresh device array copied from it
otherwise.

The mesh does this for its own metadata — `block_origins` and
`block_spacings` are uploaded inside `fill_by_coordinates!` and
`firing_boxes` — and an application has to do it for its own. It is spelled
out here rather than reaching for `TreeAMR.todevice` on purpose: this
package stands for a downstream user of the public API, and
`KernelAbstractions.allocate` is the whole of what that requires.

Its consumer here is the per-block geometry a `WaveProblem` carries, uploaded
once when the problem is built and read by the right-hand-side kernel at
every stage of every step.
"""
to_backend(::CPU, a::AbstractArray) = a

function to_backend(backend::Backend, a::AbstractArray)
    dev = allocate(backend, eltype(a), size(a))
    copyto!(dev, a)
    return dev
end

"""
    hostcopy(fs)

`fs` with its working array on the host: `fs` itself when it is already
there, a new field set over the same forest otherwise.

This is for the consumers that cannot be moved onto a device: anything that
reads single points through `blockview` and `coordinates` on the host.

**On the CPU it returns `fs` itself, not a copy.** Its consumers only
read, so a copy would be pure cost on the host path — which is the common
one — and returning the same object makes that explicit rather than leaving
it to a caller's `===` check. A caller that means to *write* to the result
wants [`hostcopy!`](@ref) into a field set of its own.

It is deliberately *not* how the numeric diagnostics work: the error norms
and the maximum speed go through TreeAMR's `mesh_mapreduce` and stay where
the data is.
"""
function hostcopy(fs::FieldSet{T}) where {T}
    get_backend(fs.work) isa CPU && return fs
    # The whole layout, not merely the forest: a field set carries its own
    # ghost width and centering from TreeAMR's M8 on, and either one left
    # at its default would give the copy a differently shaped working array
    # — which `hostcopy!` would then reject. `G` has no default at all,
    # which is what turns the first of those mistakes into an error message
    # instead of a wrong answer.
    host = FieldSet{T}(fs.forest, fs.nvars; G=fs.G, centering=fs.centering,
                       parity=fs.parity, rotation=fs.rotation)
    return hostcopy!(host, fs)
end

"""
    hostcopy!(dst, src)

`src`'s working array copied into `dst`'s, ghosts and all; returns `dst`.

The copying half of [`hostcopy`](@ref), split out so that it can be called
— and tested — without a device. `hostcopy` allocates the destination and
short-circuits on the host, so on a machine with no GPU the only part of it
that would otherwise run is the `return fs`; this is the other part, and a
CPU-to-CPU call exercises it for real.

The two field sets must have the same layout: the same forest, the same
`nvars`, the same per-dimension ghost widths and the same centering. That
is checked rather than assumed, because a mismatch in `G` or in the
centering changes the *shape* of the working array, and a `copyto!` between
two arrays of the same total length but different shapes would silently
transpose the data rather than fail.
"""
function hostcopy!(dst::FieldSet, src::FieldSet)
    dst.forest === src.forest || throw(ArgumentError(
        "hostcopy! needs both field sets over the same forest: the block " *
        "order and the block count come from the leaf array, so two forests " *
        "with the same leaves today would still be a different copy tomorrow."))
    size(dst.work) == size(src.work) || throw(ArgumentError(
        "hostcopy! needs the same layout on both sides, got nvars " *
        "$(dst.nvars) / $(src.nvars), G $(dst.G) / $(src.G) and centering " *
        "$(dst.centering) / $(src.centering), which store " *
        "$(size(dst.work)) and $(size(src.work)). A field set carries its " *
        "own ghost width and centering, so a destination built without them " *
        "has a differently shaped working array."))
    copyto!(dst.work, src.work)
    return dst
end
