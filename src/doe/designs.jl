module Designs

export initialize, make_initializer
export init_design, init_mixture_design
export generate_mixture_design

using LinearAlgebra
using Random
using Random
using Distributions

include("../util/tensor_ops.jl")
using .TensorOps

export make_initializer

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

end # module