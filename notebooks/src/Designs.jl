module Designs

using LinearAlgebra
using Random
using Distributions
using Polyhedra
using HiGHS
using MLStyle

using ..TensorOps
using ..HitAndRun
using ..ConstraintEnforcement


function hypercube_initializer(N, K; upper = 1, lower=-1, rng = Random.GLOBAL_RNG)
    (n) -> lower .+ rand(rng, n, N, K) .* (upper - lower)
end

# Fill an nxNxK matrix with random values ensuring each row sums to 1
function init_mixture_design(N, K, rng; n=1)
    designs = rand(rng, Dirichlet(ones(K)), N * n)
    return reshape(designs, n, N, K)
end

function mixture_initializer(N, K; rng = Random.GLOBAL_RNG)
    (n) -> init_mixture_design(N, K, rng, n = n)
end

function constrained_initializer(N, A, b; rng = Random.GLOBAL_RNG, burnin = 100)
    lib = DefaultLibrary{Float64}(HiGHS.Optimizer)
    K = size(A, 2)
    v_0 = HitAndRun.get_initial_sample(A, b, lib)
    function sample_constraints(n)
        X = HitAndRun.hit_and_run(A, b, N * n, lib; v_0 = v_0, burnin = burnin, rng = rng)
        return reshape(X, (n, N, K))
    end
    return sample_constraints
end

function create_initializer(constraints::ConstraintEnforcement.Constraints, N::Int64, K::Int64; rng = Random.GLOBAL_RNG)
    @match constraints begin
        ConstraintEnforcement.LinearConstraints(A, b) => Designs.constrained_initializer(N, constraints.A, constraints.b; rng = rng)
        ConstraintEnforcement.NoConstraints() => Designs.hypercube_initializer(N, K; rng = rng)
        _ => error("Unsupported constraint type: $(typeof(constraints))")
    end
end

end # module