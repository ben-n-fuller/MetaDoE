module Designs

export initialize, make_initializer
export init_design, init_mixture_design
export generate_mixture_design

using LinearAlgebra
using Random
using Distributions
using Polyhedra
using HiGHS

using ..TensorOps
using ..HitAndRun
using .TensorOps


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
    function sample_constraints(n)
        X = HitAndRun.hit_and_run(A, b, N * n, lib; burnin = burnin, rng = rng)
        return reshape(X, (n, N, K))
    end
    return sample_constraints
end

end # module