# Checkpoint and restart, through TreeIOHDF5 (`CODE.md`, "Refinement and the
# driver").
#
# A checkpoint is written at a chunk boundary, after the regrid: a fixed-step
# integrator then holds nothing but `(t, u)`, and `t` is a function of the
# chunk index, so the mesh, the state vector and the chunk index are the
# whole run. The case, the order and the criterion are the caller's, and a
# restart is the same `evolve!` call with `restart` set; the order and the
# block size are checked against the file.

const APPLICATION = "TreeWaveGR"
const FORMAT_VERSION = 1

"""
    save_run(path, forest, fs, u; chunk, nsteps, nregrids, passes = 0, q,
             sync = true)

Write the run after chunk `chunk` to `path`: the mesh, the state `u` of
`fs`, and the run's counters (`passes` is the initial-data cycle's).
"""
function save_run(path::AbstractString, forest::Forest, fs::FieldSet, u; chunk::Integer,
                  nsteps::Integer, nregrids::Integer, passes::Integer=0, q::Integer,
                  sync::Bool=true)
    save_checkpoint(path, forest; fieldsets=("state" => (fs, u),),
                    application=APPLICATION => FORMAT_VERSION,
                    data=(; chunk=Int(chunk), nsteps=Int(nsteps), nregrids=Int(nregrids),
                          passes=Int(passes), q=Int(q)), sync=sync)
    return path
end

"""
    load_run(path; backend = CPU(), types = ()) -> (; forest, fs, u, data)

Read a run written by [`save_run`](@ref). `types` names the element types
TreeIOHDF5 cannot name on its own, such as `(Float32x2,)`. The field set's
ghosts are not filled; the next right-hand side fills them.
"""
function load_run(path::AbstractString; backend=CPU(), types=())
    ck = load_checkpoint(path; backend=backend, types=types)
    name, version = ck.application
    name == APPLICATION || throw(ArgumentError(
        "$path is a checkpoint of $name, not of $APPLICATION"))
    version == FORMAT_VERSION || throw(ArgumentError(
        "$path is in $APPLICATION's format $version; this version reads $FORMAT_VERSION"))
    st = ck.fieldsets["state"]
    return (; forest=ck.forest, fs=st.fieldset, u=st.state, data=ck.data)
end

"""
    checkpoint_path(prefix, chunk) -> String

The file of the checkpoint after chunk `chunk`: `prefix.chunk000012.h5`.
"""
checkpoint_path(prefix::AbstractString, chunk::Integer) =
    string(prefix, ".chunk", lpad(chunk, 6, '0'), ".h5")
