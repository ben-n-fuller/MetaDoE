module Objectives

include("../util/tensor_ops.jl")
using .TensorOps

using SpecialFunctions
using LinearAlgebra

function d_criterion(X::Array{Float64, 2})
    score = abs(det(X' * X))
    return score == 0 ? Inf : 1 / score
end

function D(X::Array{Float64, 3})
    TensorOps.squeeze(mapslices(d_criterion, X, dims=[2, 3]))
end

function rastrigin(X::Array{Float64, 3})
    A = 10
    return TensorOps.squeeze(A * size(X,2) * size(X,3) .+ sum(X.^2 .- A * cos.(2π * X), dims=(2,3)))
end

function griewank(X::Array{Float64, 3})
    N, K = size(X,2), size(X,3)
    d = N * K
    j = reshape(1:d, (1, N, K))

    term1 = sum(X.^2, dims=(2,3)) / 4000  # Polynomial term
    term2 = prod(cos.(X ./ sqrt.(j)), dims=(2,3))  # Product term
    return TensorOps.squeeze(term1 .- term2 .+ 1)
end

end # module