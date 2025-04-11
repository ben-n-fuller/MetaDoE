module Constraints

using LinearAlgebra

function hypercube(n::Int, k::Real)
    I_n = Matrix{Float64}(I, n, n)
    A = vcat(I_n, -I_n)
    b = vcat(fill(Float64(k), n), fill(Float64(k), n))
    return A, b
end

function simplex(K)
    # Non-negativity
    A_nonneg = -I(K)

    # Sum-to-one
    A_sum_upper = ones(1, K)
    A_sum_lower = -ones(1, K)

    # Combine constraints
    A = vcat(A_nonneg, A_sum_upper, A_sum_lower)
    b = vcat(zeros(K), [1.0], [-1.0])

    return A, b
end

end # module