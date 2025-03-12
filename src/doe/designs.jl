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

# Walsh implementation for generating mixture designs
function generate_mixture_design(N, K)
    # N:= number of points in design (nrows)
    # K:= number of factors (ncols)
    # generate from uniform dirichlet
    alpha = ones(K)
    d     = Dirichlet(alpha)
    X     = transpose(rand(d, N))
    # make sure smallest is bigger than 0
    smallest = eps()^2
    for i in 1:N
        xt = X[i, :]
        xt[xt .< smallest] .= smallest
        xt = xt/sum(xt)
        X[i, :] = xt
    end

    return X
end

function hypercube_initializer(N, K; rng = Random.GLOBAL_RNG)
    (n) -> lower .+ rand(rng, n, N, K) .* (upper - lower)
end

# Fill an nxNxK matrix with values sampled from a uniform dist on [lower, upper]
function init_design(N, K, rng; n=1, lower=-1, upper=1)
    lower .+ rand(rng, n, N, K) .* (upper - lower)
end

function initialize_hypercube(N, K; rng = Random.GLOBAL_RNG)
    return (n) -> init_design(N, K, rng; n = n)
end

# Fill an nxNxK matrix with random values ensuring each row sums to 1
function init_mixture_design(N, K, rng; n=1)
    designs = rand(rng, Dirichlet(ones(K)), N * n)
    return reshape(designs, n, N, K)
end

function fill_invalid!(X, model_builder, init)
    _, N, K = size(X)

    M = (TensorOps.expand ∘ model_builder)(X)

    check_invalid = (x) -> rank(x) < K
    invalids = mapslices(check_invalid, M, dims=[2,3])
    invalid_indices = (findall ∘ TensorOps.squeeze)(invalids)

    # If no invalid designs, return
    if length(invalid_indices) == 0
        return X
    end

    # Replace invalid designs with new ones in-place
    X[invalid_indices, :, :] = init(N, K, length(invalid_indices))

    # Recursively fill invalid designs
    return fill_invalid!(X, model_builder, init)
end

function init_filtered_design(N, K, model_builder; n = 1, init = init_design)
    # Initialize designs
    designs = init(N, K, n)

    # Filter out invalid designs
    fill_invalid!(designs, model_builder, init)
    return designs
end

function initialize(N, K, model_builder; n = 1, type="uniform", rng = Random.GLOBAL_RNG)
    init_func = type == "uniform" ? (N, K, n) -> init_design(N, K, rng; n=n) : (N, K, n) -> init_mixture_design(N, K, rng; n=n)
    return init_filtered_design(N, K, model_builder; n = n, init = init_func)
end

function make_initializer(N, K, model_builder; type="uniform", rng = Random.GLOBAL_RNG)
    return (n) -> initialize(N, K, model_builder, n = n, type=type, rng = rng)
end

end