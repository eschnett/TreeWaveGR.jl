# The cost of one right-hand-side evaluation, per owned point.
#
#     julia --project=. -t 8 bench/rhs.jl [--type=f64|f32] [--N=16] [--roots=4]
#     julia --project=<env with Metal> bench/rhs.jl --backend=metal --type=f32
#
# On the static and the boosted hole (a = 0.6), on the box [3, 7]×[−2, 2]²
# outside the horizon, at q = 4 with dissipation, for every coefficient
# source the background allows: the whole `wave_rhs!` (scatter, ghost fill,
# kernel) and the kernel alone (`launch_rhs!`). Best of `reps`.

using SpacetimeMetrics
using TreeAMR
using TreeWaveGR
using KernelAbstractions

args = Dict(split(a[3:end], '=')[1] => split(a[3:end], '=')[2] for a in ARGS
            if startswith(a, "--") && occursin('=', a))
const BACKEND_NAME = get(args, "backend", "cpu")
if BACKEND_NAME == "metal"
    using Metal
elseif BACKEND_NAME == "cuda"
    using CUDA
end
backend = BACKEND_NAME == "metal" ? Metal.MetalBackend() :
          BACKEND_NAME == "cuda" ? CUDA.CUDABackend() : CPU()
T = get(args, "type", "f64") == "f32" ? Float32 : Float64
N = parse(Int, get(args, "N", "16"))
roots = parse(Int, get(args, "roots", "4"))
reps = parse(Int, get(args, "reps", "10"))

function best(f, reps)
    f()
    return minimum(_ -> (t0 = time_ns(); f(); time_ns() - t0), 1:reps)
end

for (name, m, sol) in [("static hole", KerrSchild(1.0, 0.6), StaticHole(1.0, 0.6)),
                       ("boosted hole", boost(KerrSchild(1.0, 0.6), [0.6, 0, 0]),
                        StaticHole(1.0, 0.6))]
    case = WaveCase(Background{3}(m), sol; extents=((3.0, 7.0), (-2.0, 2.0), (-2.0, 2.0)),
                    ε=0.25)
    forest = wave_forest(T, case; N=N, roots=roots)
    fs = state_fieldset(forest, 4; backend=backend)
    sources = TreeWaveGR.is_stationary(m) ? (:sampled, :analytic) : (:analytic,)
    for coefficients in sources
        p = WaveProblem(fs, GhostSchedule(fs, wave_operators(4)), case; q=4,
                        coefficients=coefficients)
        u = exact_statevector(fs, case, 0)
        du = similar(u)
        npoints = N^3 * nblocks(fs)
        full = best(() -> wave_rhs!(du, u, p, 0), reps)
        prepare_rhs!(p, u, 0)
        kernel = best(() -> launch_rhs!(du, p, 0), reps)
        println(rpad("$name, $coefficients", 26), " $T on $backend, $(Threads.nthreads()) ",
                "thread(s), $npoints points: wave_rhs! $(round(full / npoints; digits=2)) ns, ",
                "kernel $(round(kernel / npoints; digits=2)) ns per point")
    end
end
