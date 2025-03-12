module OptimalityCriteria

include("../util/tensor_ops.jl")
using .TensorOps

export g_criterion, d_criterion

using SpecialFunctions
using LinearAlgebra

function d_criterion(X::Array{Float64, 2})
    score = abs(det(X' * X))
    return score == 0 ? Inf : 1 / score
end

function D(X::Array{Float64, 3})
    TensorOps.squeeze(mapslices(d_criterion, X, dims=[2, 3]))
end

end