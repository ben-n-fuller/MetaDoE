module Objectives

using ..TensorOps

using SpecialFunctions
using LinearAlgebra

function D_mat(F; ε=1e-6)
    # return 1/det(permutedims(F, (2, 1)) * F)


    # N, p = size(F)
    # M = F' * F

    # If N >= p, we can use the Cholesky decomposition
    # if N >= p 
    #     L = cholesky(M).L
    #     return -sum(log.(diag(L)))
    # end

    # If N < p, use SVD
    svals = svdvals(F)
    return -sum(log.(svals[svals .> ε]))
end

function A_mat(F; ε=1e-6)
    N, p = size(F)

    if N >= p
        L = cholesky(F' * F).L
        Linv = inv(L)
        return norm(Linv, 2)^2
    end

    # If N < p, use SVD
    svals = svdvals(F)
    return sum(1 ./ (svals[svals .> ε].^2))
end


function A(F)
    map(A_mat, eachslice(F; dims=1))
end

function D(F)
    map(D_mat, eachslice(F; dims=1))
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

function rosenbrock(X::Array{Float64, 3})
    X1 = X[:, :, 1:end-1]
    X2 = X[:, :, 2:end]
    term1 = 100 .* (X2 .- X1.^2).^2
    term2 = (1 .- X1).^2
    return TensorOps.squeeze(sum(term1 .+ term2, dims=(2,3)))
end

function ackley(X::Array{Float64, 3})
    N, K = size(X,2), size(X,3)
    d = N * K

    sum_sq = sum(X.^2, dims=(2,3))
    cos_term = sum(cos.(2π .* X), dims=(2,3))

    term1 = -20 .* exp.(-0.2 .* sqrt.(sum_sq ./ d))
    term2 = -exp.(cos_term ./ d)

    return TensorOps.squeeze(term1 .+ term2 .+ 20 .+ ℯ)
end


end # module